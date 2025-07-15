# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#			http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pytorch version of patched decoder."""

import dataclasses
import math
from typing import List, Tuple, Optional
import torch
from torch import nn
import torch.nn.functional as F


def _create_quantiles() -> list[float]:
	return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@dataclasses.dataclass
class TimesFMConfig:
	"""Config for initializing timesfm patched_decoder class."""

	# The number of blocks in the model.
	num_layers: int = 20
	# The number of attention heads used in the attention layers of the model.
	num_heads: int = 16
	# The number of key-value heads for implementing attention.
	num_kv_heads: int = 16
	# The hidden size of the model.
	hidden_size: int = 1280
	# The dimension of the MLP representations.
	intermediate_size: int = 1280
	# The number of head dimensions.
	head_dim: int = 80
	# The epsilon used by the rms normalization layers.
	rms_norm_eps: float = 1e-6
	# Patch length
	patch_len: int = 32
	# Horizon length
	horizon_len: int = 128
	# quantiles
	quantiles: List[float] = dataclasses.field(default_factory=_create_quantiles)
	# Padding value
	pad_val: float = 1123581321.0
	# Tolerance
	tolerance: float = 1e-6
	# The dtype of the weights.
	dtype: str = "bfloat32"
	# use positional embedding
	use_positional_embedding: bool = True


def _masked_mean_std(
		inputs: torch.Tensor,
		padding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	"""Calculates mean and standard deviation of `inputs` across axis 1.

	It excludes values where `padding` is 1.

	Args:
		inputs: A PyTorch tensor of shape [b, n, p].
		padding: A PyTorch tensor of shape [b, n, p] with values 0 or 1.

	Returns:
		A tuple containing the mean and standard deviation.
		We return the statistics of the first patch with more than three non-padded
		values.
	"""
	# Selecting the first patch with more than 3 unpadded values.
	pad_sum = torch.sum(1 - padding, dim=2)

	def _get_patch_index(arr: torch.Tensor):
		indices = torch.argmax((arr >= 3).to(torch.int32), dim=1)
		row_sum = (arr >= 3).to(torch.int32).sum(dim=1)
		return torch.where(row_sum == 0, arr.shape[1] - 1, indices)

	patch_indices = _get_patch_index(pad_sum)
	bidxs = torch.arange(inputs.shape[0])

	arr = inputs[bidxs, patch_indices, :]
	pad = padding[bidxs, patch_indices, :]

	# Create a mask where padding is 0
	mask = 1 - pad

	# Calculate the number of valid elements
	num_valid_elements = torch.sum(mask, dim=1)
	num_valid_elements = torch.where(
			num_valid_elements == 0,
			torch.tensor(1,
									 dtype=num_valid_elements.dtype,
									 device=num_valid_elements.device),
			num_valid_elements,
	)

	# Calculate the masked sum and squared sum
	masked_sum = torch.sum(arr * mask, dim=1)
	masked_squared_sum = torch.sum((arr * mask)**2, dim=1)

	# Calculate the masked mean and standard deviation
	masked_mean = masked_sum / num_valid_elements
	masked_var = masked_squared_sum / num_valid_elements - masked_mean**2
	masked_var = torch.where(
			masked_var < 0.0,
			torch.tensor(0.0, dtype=masked_var.dtype, device=masked_var.device),
			masked_var,
	)
	masked_std = torch.sqrt(masked_var)

	return masked_mean, masked_std


def _shift_padded_seq(mask: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
	"""Shifts rows of seq based on the first 0 in each row of the mask.

	Args:
		mask: mask tensor of shape [B, N]
		seq: seq tensor of shape [B, N, P]

	Returns:
		Returns the shifted sequence.
	"""
	batch_size, num_seq, feature_dim = seq.shape

	new_mask: torch.BoolTensor = mask == 0

	# Use argmax to find the first True value in each row
	indices = new_mask.to(torch.int32).argmax(dim=1)

	# Handle rows with all zeros
	indices[~new_mask.any(dim=1)] = -1

	# Create index ranges for each sequence in the batch
	idx_range = (torch.arange(num_seq).to(seq.device).unsqueeze(0).unsqueeze(-1).expand(batch_size, -1,feature_dim))

	# Calculate shifted indices for each element in each sequence
	shifted_idx = (idx_range - indices[:, None, None]) % num_seq

	# Gather values from seq using shifted indices
	shifted_seq = seq.gather(1, shifted_idx)

	return shifted_seq


def get_large_negative_number(dtype: torch.dtype) -> torch.Tensor:
	"""Returns a large negative value for the given dtype."""
	if dtype.is_floating_point:
		dtype_max = torch.finfo(dtype).max
	else:
		dtype_max = torch.iinfo(dtype).max
	return torch.tensor(-0.7 * dtype_max, dtype=dtype)


def apply_mask_to_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
	"""Applies a floating-point mask to a set of logits.

	Args:
			logits: A torch.Tensor of logit values.
			mask: A torch.Tensor (float32) of mask values with the encoding described
				in the function documentation.

	Returns:
			Masked logits.
	"""

	min_value = get_large_negative_number(logits.dtype)

	return torch.where((mask >= min_value * 0.5), logits, min_value)


def convert_paddings_to_mask(
		paddings: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
	"""Converts binary paddings to a logit mask ready to add to attention matrix.

	Args:
			paddings: binary torch.Tensor of shape [B, T], with 1 denoting padding
				token.
			dtype: data type of the input.

	Returns:
			A torch.Tensor of shape [B, 1, 1, T] ready to add to attention logits.
	"""
	attention_mask = paddings.detach().clone()
	attention_mask = attention_mask[:, None, None, :]	 # Equivalent to jnp.newaxis
	attention_mask *= get_large_negative_number(dtype)
	return attention_mask


def causal_mask(input_t: torch.Tensor) -> torch.Tensor:
	"""Computes and returns causal mask.

	Args:
			input_t: A torch.Tensor of shape [B, T, D].

	Returns:
			An attention_mask torch.Tensor of shape [1, 1, T, T]. Attention mask has
			already been converted to large negative values.
	"""
	assert input_t.dtype.is_floating_point, input_t.dtype
	large_negative_number = get_large_negative_number(input_t.dtype)
	t = input_t.shape[1]
	col_idx = torch.arange(t).unsqueeze(0).repeat(t, 1)
	row_idx = torch.arange(t).unsqueeze(1).repeat(1, t)
	mask = (row_idx < col_idx).to(input_t.dtype) * large_negative_number
	return (mask.unsqueeze(0).unsqueeze(0).to(input_t.device)
				 )	# Equivalent to jnp.newaxis


def merge_masks(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
	"""Merges 2 masks.

	logscale mask is expected but 0/1 mask is also fine.

	Args:
			a: torch.Tensor of shape [1|B, 1, 1|T, S].
			b: torch.Tensor of shape [1|B, 1, 1|T, S].

	Returns:
			torch.Tensor of shape [1|B, 1, 1|T, S].
	"""

	def expand_t(key_mask):
		query_mask = key_mask.transpose(-1, -2)	 # Equivalent of jnp.transpose
		return torch.minimum(query_mask, key_mask)

	if a.shape[2] != b.shape[2]:
		if a.shape[2] == 1:
			a = expand_t(a)
		else:
			assert b.shape[2] == 1
			b = expand_t(b)

	assert a.shape[1:] == b.shape[1:], f"a.shape={a.shape}, b.shape={b.shape}."
	return torch.minimum(a, b)	# Element-wise minimum, similar to jnp.minimum


class ResidualBlock(nn.Module):
	"""TimesFM residual block."""

	def __init__(
			self,
			input_dims,
			hidden_dims,
			output_dims,
	):
		super(ResidualBlock, self).__init__()
		self.input_dims = input_dims
		self.hidden_dims = hidden_dims
		self.output_dims = output_dims

		# Hidden Layer
		self.hidden_layer = nn.Sequential(
				nn.Linear(input_dims, hidden_dims),
				nn.SiLU(),
		)

		# Output Layer
		self.output_layer = nn.Linear(hidden_dims, output_dims)
		# Residual Layer
		self.residual_layer = nn.Linear(input_dims, output_dims)

	def forward(self, x):
		hidden = self.hidden_layer(x)
		output = self.output_layer(hidden)
		residual = self.residual_layer(x)
		return output + residual


class RMSNorm(torch.nn.Module):
	"""Pax rms norm in pytorch."""

	def __init__(
			self,
			dim: int,
			eps: float = 1e-6,
			add_unit_offset: bool = False,
	):
		super().__init__()
		self.eps = eps
		self.add_unit_offset = add_unit_offset
		self.weight = nn.Parameter(torch.zeros(dim))

	def _norm(self, x):
		return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

	def forward(self, x):
		output = self._norm(x.float())
		if self.add_unit_offset:
			output = output * (1 + self.weight.float())
		else:
			output = output * self.weight.float()
		return output.type_as(x)


class TransformerMLP(nn.Module):
	"""Pax transformer MLP in pytorch."""

	def __init__(
			self,
			hidden_size: int,
			intermediate_size: int,
	):
		super().__init__()
		self.gate_proj = nn.Linear(hidden_size, intermediate_size)
		self.down_proj = nn.Linear(intermediate_size, hidden_size)
		self.layer_norm = nn.LayerNorm(normalized_shape=hidden_size, eps=1e-6)

	def forward(self, x, paddings=None):
		gate_inp = self.layer_norm(x)
		gate = self.gate_proj(gate_inp)
		gate = F.relu(gate)
		outputs = self.down_proj(gate)
		if paddings is not None:
			outputs = outputs * (1.0 - paddings[:, :, None])
		return outputs + x


class TimesFMAttention(nn.Module):
		"""Implements the attention used in TimesFM."""

		def __init__(
				self,
				hidden_size: int,
				num_heads: int,
				num_kv_heads: int,
				head_dim: int,
		):
				super().__init__()

				self.num_heads = num_heads
				self.num_kv_heads = num_kv_heads

				assert self.num_heads % self.num_kv_heads == 0
				self.num_queries_per_kv = self.num_heads // self.num_kv_heads

				self.hidden_size = hidden_size
				self.head_dim = head_dim

				self.q_size = self.num_heads * self.head_dim
				self.kv_size = self.num_kv_heads * self.head_dim
				self.scaling = nn.Parameter(
						torch.empty((self.head_dim,), dtype=torch.float32),
				)

				self.qkv_proj = nn.Linear(
						self.hidden_size,
						(self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
				)
				self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)

		def _per_dim_scaling(self, query: torch.Tensor) -> torch.Tensor:
				# [batch_size, n_local_heads, input_len, head_dim]
				r_softplus_0 = 1.442695041
				softplus_func = torch.nn.Softplus()
				scale = r_softplus_0 / math.sqrt(self.head_dim)
				scale = scale * softplus_func(self.scaling)
				return query * scale[None, None, None, :]

		def forward(
				self,
				hidden_states: torch.Tensor,
				mask: torch.Tensor,
				kv_states: torch.Tensor | None = None,
				kv_write_indices: torch.Tensor | None = None,
				kv_cache: Tuple[torch.Tensor, torch.Tensor] | None = None,
		) -> torch.Tensor:
				hidden_states_shape = hidden_states.shape
				assert len(hidden_states_shape) == 3
				batch_size, input_len, _ = hidden_states_shape

				# Split qkv_proj weights and biases
				qkv_weight = self.qkv_proj.weight
				q_weight = qkv_weight[:self.q_size, :]
				k_weight = qkv_weight[self.q_size : self.q_size + self.kv_size, :]
				v_weight = qkv_weight[-self.kv_size:, :]

				if self.qkv_proj.bias is not None:
					qkv_bias = self.qkv_proj.bias
					q_bias = qkv_bias[:self.q_size]
					k_bias = qkv_bias[self.q_size : self.q_size + self.kv_size]
					v_bias = qkv_bias[-self.kv_size:]
				else:
					q_bias = k_bias = v_bias = None

				# Compute queries from hidden_states
				xq = F.linear(hidden_states, q_weight, q_bias)
				xq = xq.view(batch_size, input_len, self.num_heads, self.head_dim)
				xq = self._per_dim_scaling(xq)

				if kv_states is None:
						# Compute keys and values from hidden_states (original behavior)
						xk = F.linear(hidden_states, k_weight, k_bias)
						xv = F.linear(hidden_states, v_weight, v_bias)
						xk = xk.view(batch_size, input_len, self.num_kv_heads, self.head_dim)
						xv = xv.view(batch_size, input_len, self.num_kv_heads, self.head_dim)
						seq_len = input_len
				else:
						# Compute keys and values from kv_states
						kv_states_shape = kv_states.shape
						assert len(kv_states_shape) == 3 and kv_states_shape[0] == batch_size and kv_states_shape[2] == self.hidden_size
						seq_len = kv_states_shape[1]
						xk = F.linear(kv_states, k_weight, k_bias)
						xv = F.linear(kv_states, v_weight, v_bias)
						xk = xk.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
						xv = xv.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

				# Handle kv_cache
				if kv_cache is not None and kv_write_indices is not None:
						k_cache, v_cache = kv_cache
						k_cache.index_copy_(1, kv_write_indices, xk)
						v_cache.index_copy_(1, kv_write_indices, xv)
						key = k_cache
						value = v_cache
				else:
						key = xk
						value = xv

				if self.num_kv_heads != self.num_heads:
						key = torch.repeat_interleave(key, self.num_queries_per_kv, dim=2)
						value = torch.repeat_interleave(value, self.num_queries_per_kv, dim=2)

				q = xq.transpose(1, 2)	# [batch_size, num_heads, input_len, head_dim]
				k = key.transpose(1, 2)	 # [batch_size, num_heads, seq_len, head_dim]
				v = value.transpose(1, 2)	 # [batch_size, num_heads, seq_len, head_dim]

				scores = torch.matmul(q, k.transpose(2, 3))	 # [batch_size, num_heads, input_len, seq_len]

				# Adjust mask for cross-attention if kv_states is provided
				if kv_states is not None:
						# Create a new mask for cross-attention
						# Assuming no padding in kv_states for simplicity
						mask = torch.zeros(
								batch_size, 1, input_len, seq_len,
								dtype=hidden_states.dtype, device=hidden_states.device
						)

				scores = scores + mask	# Now mask should match scores shape
				scores = F.softmax(scores.float(), dim=-1).type_as(q)

				output = torch.matmul(scores, v)	# [batch_size, num_heads, input_len, head_dim]
				output = output.transpose(1, 2).contiguous().view(batch_size, input_len, -1)
				output = self.o_proj(output)
				return scores, output


class TimesFMDecoderLayer(nn.Module):
	"""Transformer layer."""

	def __init__(
			self,
			hidden_size: int,
			intermediate_size: int,
			num_heads: int,
			num_kv_heads: int,
			head_dim: int,
			rms_norm_eps: float = 1e-6,
	):
		super().__init__()
		self.self_attn = TimesFMAttention(
				hidden_size=hidden_size,
				num_heads=num_heads,
				num_kv_heads=num_kv_heads,
				head_dim=head_dim,
		)
		self.mlp = TransformerMLP(
				hidden_size=hidden_size,
				intermediate_size=intermediate_size,
		)
		self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
		self.g_l = nn.Parameter(torch.tensor(0.))

	def forward(
			self,
			hidden_states: torch.Tensor,
			mask: torch.Tensor,
			paddings: torch.Tensor,
			kv_write_indices: torch.Tensor | None = None,
			kv_cache: Tuple[torch.Tensor, torch.Tensor] | None = None,
			additional_token: List[torch.Tensor] | None = None,
	) -> torch.Tensor:
		hidden_states_addn = hidden_states
		residual = hidden_states
		# Self Attention
		hidden_states = self.input_layernorm(hidden_states)
		scores, hidden_states = self.self_attn(
				hidden_states=hidden_states,
				mask=mask,
				kv_write_indices=kv_write_indices,
				kv_cache=kv_cache,
		)
		hidden_states = hidden_states + residual
		# Additional context
		if additional_token is not None:
			_, hidden_states_addn = self.self_attn(
					hidden_states=hidden_states_addn,
					mask=mask,
					kv_states=additional_token,
					kv_write_indices=kv_write_indices,
					kv_cache=kv_cache,
			)
			hidden_states_addn = hidden_states_addn
			hidden_states = hidden_states + F.tanh(self.g_l) * hidden_states_addn

		# print(f"From inside the MHA, residual shape is {residual.shape} and hidden_states shape is {hidden_states.shape}.")

		# MLP
		hidden_states = self.mlp(hidden_states, paddings=paddings)
		return scores, hidden_states


class StackedDecoder(nn.Module):
	"""Stacked transformer layer."""

	def __init__(
			self,
			hidden_size: int,
			intermediate_size: int,
			num_heads: int,
			num_kv_heads: int,
			head_dim: int,
			num_layers: int,
			rms_norm_eps: float = 1e-6,
	):
		super().__init__()

		self.layers = nn.ModuleList()
		for _ in range(num_layers):
			self.layers.append(
					TimesFMDecoderLayer(
							hidden_size=hidden_size,
							intermediate_size=intermediate_size,
							num_heads=num_heads,
							num_kv_heads=num_kv_heads,
							head_dim=head_dim,
							rms_norm_eps=rms_norm_eps,
					))

	def forward(
			self,
			hidden_states: torch.Tensor,
			paddings: torch.Tensor,
			kv_write_indices: torch.Tensor | None = None,
			kv_caches: List[Tuple[torch.Tensor, torch.Tensor]] | None = None,
			additional_tokens: List[torch.Tensor] | None = None
	) -> torch.Tensor:
		padding_mask = convert_paddings_to_mask(paddings, hidden_states.dtype)
		atten_mask = causal_mask(hidden_states)
		mask = merge_masks(padding_mask, atten_mask)
		len_add_tokens = len(additional_tokens) if additional_tokens is not None else None
		num_layers = len(self.layers)
		for i in range(num_layers):
			if len_add_tokens is not None and i >= num_layers - len_add_tokens:
				additional_token = additional_tokens[i-(num_layers-len_add_tokens)]
			else:
				additional_token = None
			layer = self.layers[i]
			kv_cache = kv_caches[i] if kv_caches is not None else None
			_, hidden_states = layer(
					hidden_states=hidden_states,
					mask=mask,
					paddings=paddings,
					kv_write_indices=kv_write_indices,
					kv_cache=kv_cache,
					additional_token=additional_token
			)
		return hidden_states


class PositionalEmbedding(torch.nn.Module):
	"""Generates position embedding for a given 1-d sequence.

	Attributes:
			min_timescale: Start of the geometric index. Determines the periodicity of
				the added signal.
			max_timescale: End of the geometric index. Determines the frequency of the
				added signal.
			embedding_dims: Dimension of the embedding to be generated.
	"""

	def __init__(
			self,
			embedding_dims: int,
			min_timescale: int = 1,
			max_timescale: int = 10_000,
	) -> None:
		super().__init__()
		self.min_timescale = min_timescale
		self.max_timescale = max_timescale
		self.embedding_dims = embedding_dims

	def forward(self, seq_length=None, position=None):
		"""Generates a Tensor of sinusoids with different frequencies.

		Args:
				seq_length: an optional Python int defining the output sequence length.
					if the `position` argument is specified.
				position:		[B, seq_length], optional position for each token in the
					sequence, only required when the sequence is packed.

		Returns:
				[B, seqlen, D] if `position` is specified, else [1, seqlen, D]
		"""
		if position is None:
			assert seq_length is not None
			# [1, seqlen]
			position = torch.arange(seq_length, dtype=torch.float32).unsqueeze(0)
		else:
			assert position.ndim == 2, position.shape

		num_timescales = self.embedding_dims // 2
		log_timescale_increment = math.log(
				float(self.max_timescale) / float(self.min_timescale)) / max(
						num_timescales - 1, 1)
		inv_timescales = self.min_timescale * torch.exp(
				torch.arange(num_timescales, dtype=torch.float32) *
				-log_timescale_increment)
		scaled_time = position.unsqueeze(2) * inv_timescales.unsqueeze(0).unsqueeze(
				0)
		signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=2)
		# Padding to ensure correct embedding dimension
		signal = F.pad(signal, (0, 0, 0, self.embedding_dims % 2))
		return signal
	
class SinusoidalEmbedding(nn.Module):
	def __init__(self, d_model: int, max_len: int = 5000):
		super().__init__()
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		self.register_buffer('pe', pe)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		x: (B, T, C) or (T, C) or (T,)
		returns: pe slice of shape (T, C)
		"""
		if x.dim() == 3:
			T = x.size(1)
		else:
			T = x.size(0)
		return self.pe[:T]


class PatchedTimeSeriesDecoder(nn.Module):
	"""Patched time-series decoder."""

	def __init__(self, config: TimesFMConfig):
		super().__init__()
		self.config = config
		self.input_ff_layer = ResidualBlock(
				input_dims=2 * config.patch_len,
				output_dims=config.hidden_size,
				hidden_dims=config.intermediate_size,
		)
		self.freq_emb = nn.Embedding(num_embeddings=3,
																 embedding_dim=config.hidden_size)
		self.horizon_ff_layer = ResidualBlock(
				input_dims=config.hidden_size,
				output_dims=config.horizon_len * (1 + len(config.quantiles)),
				hidden_dims=config.intermediate_size,
		)
		self.stacked_transformer = StackedDecoder(
				hidden_size=self.config.hidden_size,
				intermediate_size=self.config.intermediate_size,
				num_heads=self.config.num_heads,
				num_kv_heads=self.config.num_kv_heads,
				head_dim=self.config.head_dim,
				num_layers=self.config.num_layers,
				rms_norm_eps=self.config.rms_norm_eps,
		)
		if self.config.use_positional_embedding:
			self.position_emb = PositionalEmbedding(self.config.hidden_size)

	def _forward_transform(
			self, inputs: torch.Tensor, patched_pads: torch.Tensor
	) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
		"""Input is of shape [B, N, P]."""
		# print(f"Line 636: inputs shape in {inputs.shape}, patched_pads shape is {patched_pads.shape}.")
		mu, sigma = _masked_mean_std(inputs, patched_pads)
		sigma = torch.where(
				sigma < self.config.tolerance,
				torch.tensor(1.0, dtype=sigma.dtype, device=sigma.device),
				sigma,
		)

		# Normalize each patch
		outputs = (inputs - mu[:, None, None]) / sigma[:, None, None]
		outputs = torch.where(
				torch.abs(inputs - self.config.pad_val) < self.config.tolerance,
				torch.tensor(self.config.pad_val,
							dtype=outputs.dtype,
							device=outputs.device),
				outputs,
		)
		return outputs, (mu, sigma)

	def _reverse_transform(
			self, outputs: torch.Tensor, stats: tuple[torch.Tensor,torch.Tensor]) -> torch.Tensor:
		"""Output is of shape [B, N, P, Q]."""
		mu, sigma = stats
		return outputs * sigma[:, None, None, None] + mu[:, None, None, None]

	def _preprocess_input(
			self,
			input_ts: torch.Tensor,
			input_padding: torch.Tensor,
	) -> tuple[
			torch.Tensor,
			torch.Tensor,
			tuple[torch.Tensor, torch.Tensor] | None,
			torch.Tensor,
			torch.Tensor | None
	]:
		"""Preprocess input for stacked transformer."""

		# Reshape into patches (using view for efficiency)
		bsize = input_ts.shape[0]
		patched_inputs = input_ts.view(bsize, -1, self.config.patch_len)
		patched_pads = input_padding.view(bsize, -1, self.config.patch_len)
		# print(f"Line 678: patched_inputs shape is {patched_inputs.shape}, patched_pads shape is {patched_pads.shape}.")

		patched_inputs = torch.where(
				torch.abs(patched_pads - 1.0) < self.config.tolerance,
				torch.tensor(0.0,
							dtype=patched_inputs.dtype,
							device=patched_inputs.device),
				patched_inputs,
		)
		patched_pads = torch.where(
				torch.abs(patched_inputs - self.config.pad_val) < self.config.tolerance,
				torch.tensor(1.0, dtype=patched_pads.dtype, device=patched_pads.device),
				patched_pads,
		)
		patched_inputs, stats = self._forward_transform(patched_inputs,patched_pads)

		# B x N x D
		patched_inputs = patched_inputs * (1.0 - patched_pads)
		concat_inputs = torch.cat([patched_inputs, patched_pads], dim=-1)
		model_input = self.input_ff_layer(concat_inputs)

		# A patch should not be padded even if there is at least one zero.
		patched_padding = torch.min(patched_pads,dim=-1)[0]	# Get the values from the min result
		if self.config.use_positional_embedding:
			pos_emb = self.position_emb(model_input.shape[1]).to(model_input.device)
			pos_emb = torch.concat([pos_emb] * model_input.shape[0], dim=0)
			pos_emb = _shift_padded_seq(patched_padding, pos_emb)
			model_input += pos_emb

		# print(f"Line 708: model_input shape is {model_input.shape}.")

		return model_input, patched_padding, stats, patched_inputs

	def _postprocess_output(
			self,
			model_output: torch.Tensor,
			num_outputs: int,
			stats: tuple[torch.Tensor, torch.Tensor],
	) -> torch.Tensor:
		"""Postprocess output of stacked transformer."""

		# B x N x (H.Q)
		output_ts = self.horizon_ff_layer(model_output)

		# Reshape using view
		b, n, _ = output_ts.shape
		output_ts = output_ts.view(b, n, self.config.horizon_len, num_outputs)

		return self._reverse_transform(output_ts, stats)

	def forward(
			self,
			input_ts: torch.Tensor,
			input_padding: torch.LongTensor,
			freq: torch.Tensor,
			additional_tokens: List[torch.Tensor] | None
	) -> torch.Tensor:
		num_outputs = len(self.config.quantiles) + 1
		model_input, patched_padding, stats, _ = self._preprocess_input(
				input_ts=input_ts,
				input_padding=input_padding,
		)
		f_emb = self.freq_emb(freq)	 # B x 1 x D
		model_input += f_emb
		# print(f"Line 741: Model input shape is {model_input.shape}, f_emb shape is {f_emb.shape}, and padding shape is {patched_padding.shape}.----\n")
		model_output = self.stacked_transformer(model_input, patched_padding, additional_tokens=additional_tokens)

		output_ts = self._postprocess_output(model_output, num_outputs, stats)
		return output_ts

	def decode(
			self,
			input_ts: torch.Tensor,
			paddings: torch.Tensor,
			freq: torch.LongTensor,
			horizon_len: int,
			output_patch_len: int | None = None,
			max_len: int = 512,
			return_forecast_on_context: bool = False,
			additional_tokens: List[torch.Tensor] | None = None
	) -> tuple[torch.Tensor, torch.Tensor]:
		"""Auto-regressive decoding without caching.

		Args:
			input_ts: input time-series and paddings. Time-series shape B x C.
			paddings: padding shape B x (C + H) where H is the prediction length.
			freq: frequency shape B x 1
			horizon_len: prediction length.
			output_patch_len: output length to be fetched from one step of
				auto-regressive decoding.
			max_len: maximum training context length.
			return_forecast_on_context: whether to return the model forecast on the
				context except the first input patch.

		Returns:
			Tuple of two forecasting results:
			- Point (mean) output predictions as a tensor with shape B x H'.
			- Full predictions (mean and quantiles) as a tensor with shape
				B x H' x (1 + # quantiles).
			In particular, if return_forecast_on_context is True, H' is H plus
			the forecastable context length, i.e. context_len - (first) patch_len.
		"""
		final_out = input_ts
		context_len = final_out.shape[1]
		full_outputs = []
		if paddings.shape[1] != final_out.shape[1] + horizon_len:
			raise ValueError(
					"Length of paddings must match length of input + horizon_len:"
					f" {paddings.shape[1]} != {final_out.shape[1]} + {horizon_len}")
		if output_patch_len is None:
			output_patch_len = self.config.horizon_len
		num_decode_patches = (horizon_len + output_patch_len - 1) // output_patch_len
		for step_index in range(num_decode_patches):
			# print(f"num_decode_patches is {num_decode_patches}")
			current_padding = paddings[:, 0:final_out.shape[1]]
			input_ts = final_out[:, -max_len:]
			input_padding = current_padding[:, -max_len:]
			fprop_outputs = self(input_ts, input_padding, freq, additional_tokens=additional_tokens)
			if return_forecast_on_context and step_index == 0:
				# For the first decodings step, collect the model forecast on the
				# context except the unavailable first input batch forecast.
				new_full_ts = fprop_outputs[:, :-1, :self.config.patch_len, :]
				new_full_ts = fprop_outputs.view(new_full_ts.size(0), -1, new_full_ts.size(3))

				full_outputs.append(new_full_ts)

			# (full batch, last patch, output_patch_len, index of mean forecast = 0)
			new_ts = fprop_outputs[:, -1, :output_patch_len, 0]
			new_full_ts = fprop_outputs[:, -1, :output_patch_len, :]
			# (full batch, last patch, output_patch_len, all output indices)
			full_outputs.append(new_full_ts)
			final_out = torch.concatenate([final_out, new_ts], axis=-1)

		if return_forecast_on_context:
			# `full_outputs` indexing starts at after the first input patch.
			full_outputs = torch.concatenate(
							full_outputs,
							axis=1)[:, :(context_len - self.config.patch_len + horizon_len), :]
		else:
			# `full_outputs` indexing starts at the forecast horizon.
			full_outputs = torch.concatenate(full_outputs, axis=1)[:,0:horizon_len, :]

		return (full_outputs[:, :, 0], full_outputs)
	
class VideoTokenEmbed(nn.Module):
    """
    ↓ spatially to 16×32 → flatten → linear
    In : (B, T, C, 128, 256)   (C=5 here)
    Out: (B, T, L)             (L = token_dim)

    Peak activ-memory
        pool  : B·T·C·16·32     = 1·152·5·512   ≈ 0.4 MB (fp32)
        flat  : B·T·(C·16·32)   = 1·152·2560    ≈ 1.6 MB
        proj  : B·T·L           = 1·152·1280    ≈ 0.8 MB
    → well under 1 GB.
    """
    def __init__(self, in_channels: int = 5, token_dim: int = 1280,
                 h_out: int = 32, w_out: int = 64):
        super().__init__()
        # self.pool  = nn.AdaptiveAvgPool2d((h_out, w_out)) # data is already downsampled
        self.proj  = nn.Linear(in_channels * h_out * w_out, token_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape                 # (1,152,5,128,256)
        x = x.view(B * T, C, H, W)              # merge batch & time
        # x = self.pool(x)                        # (B*T, C,16,32)
        x = x.flatten(1)                        # (B*T, C*16*32)
        x = self.proj(x)                        # (B*T, L)
        return x.view(B, T, -1)                 # (B,T,L)
	
class PointWeatherEmbed(nn.Module):
    """
    ↓ spatially to 16×32 → flatten → linear
    In : (B, T, C, 128, 256)   (C=5 here)
    Out: (B, T, L)             (L = token_dim)

    Peak activ-memory
        pool  : B·T·C·16·32     = 1·152·5·512   ≈ 0.4 MB (fp32)
        flat  : B·T·(C·16·32)   = 1·152·2560    ≈ 1.6 MB
        proj  : B·T·L           = 1·152·1280    ≈ 0.8 MB
    → well under 1 GB.
    """
    def __init__(self, in_channels: int = 5, token_dim: int = 1280):
        super().__init__()
        # self.pool  = nn.AdaptiveAvgPool2d((h_out, w_out)) # data is already downsampled
        self.proj  = nn.Linear(in_channels, token_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape                 # (1,152,5)
        x = self.proj(x)                        # (B, T, L)
        return x                # (B,T,L)
	
def _bad_input(msg: str):
    raise ValueError(msg)
	
class TimesFM(nn.Module):

		def __init__(self, lookback: int = 512, lookahead: int = 96, context_len: int = 512,
						lora_rank = None,
						weather_model = False,
						weather_tokens_to_use = 6,
						weather_size_reduction = 4,
						weather_decoder_layers = 10,
						ckpt='/home/shourya01/timesfm/timesfm-1.0-200m-pytorch/torch_model.ckpt',
						register_grad_nan_hooks=False):

			super(TimesFM, self).__init__()
			
			self.timesfm = PatchedTimeSeriesDecoder(TimesFMConfig())
			self.lookback, self.lookahead = lookback, lookahead
			self.context_len = context_len
			self.ckpt = ckpt

			if lora_rank is not None:
				self._apply_lora_to_timesfm(rank=lora_rank)

			self.weather_model = weather_model
			if self.weather_model:
				assert weather_decoder_layers <= self.timesfm.config.num_layers, "Companion transformer cannot have more layers than TimesFM."
				self.weather_tokens_to_use = weather_tokens_to_use
				self.wsr = weather_size_reduction
				self.vt_embed = VideoTokenEmbed(in_channels=5, token_dim=self.timesfm.config.hidden_size)
				self.adapter_emb = SinusoidalEmbedding(d_model=self.timesfm.config.hidden_size, max_len = 1000)
				self.adapter_enc = nn.TransformerEncoderLayer(d_model=self.timesfm.config.hidden_size, 
												  nhead=self.timesfm.config.num_heads,
												  batch_first=True,
												  activation=F.silu)
				self.adapter = nn.TransformerEncoder(self.adapter_enc,num_layers=weather_decoder_layers)
				mask = nn.Transformer.generate_square_subsequent_mask((lookback+lookahead)//4) # weather runs 4 times slower that energy
				self.register_buffer("src_mask",mask)
				
			# # Debugging NaN grads
			# self.debug = register_grad_nan_hooks
			# if self.debug:
			# 	def nan_hook(m, inp, out):
			# 		def chk(t,name):
			# 			if isinstance(t, torch.Tensor) and (torch.isnan(t).any() or torch.isinf(t).any()):
			# 				print(f"{name} ← NaN/Inf in {m.__class__.__name__}")
			# 		for i,x in enumerate(inp):
			# 			chk(x, f"inp#{i}")
			# 		for i,x in enumerate(out if isinstance(out,(list,tuple)) else (out,)):
			# 			chk(x, f"out#{i}")
			# 	for m in self.adapter.modules():
			# 		m.register_forward_hook(nan_hook)

		def load_weights(self, print_missing_unexpected: bool = False):

			if self.ckpt is not None:
				results = self.timesfm.load_state_dict(torch.load(self.ckpt,map_location="cpu"),strict=False)
				if print_missing_unexpected:
					print(f"Results of loading weights: {results}")
			else:
				raise ValueError("Checkpoint path not provided.")
			
		def _apply_lora_to_timesfm(self, rank: int = 4, alpha: float = 1.0):
			# …freeze params as before…
			for n, p in self.timesfm.named_parameters():
				p.requires_grad = (n.split('.')[-1] == 'g_l')

			def _replace(root, name, new_mod):
				parts = name.split('.')
				parent = root
				for p in parts[:-1]:
					parent = getattr(parent, p)
				setattr(parent, parts[-1], new_mod)

			class LoRALinear(nn.Module):
				def __init__(self, orig: nn.Linear):
					super().__init__()
					self.in_features = orig.in_features
					self.out_features = orig.out_features
					self.bias = orig.bias
					self.weight = orig.weight
					self.weight.requires_grad = False
					self.A = nn.Parameter(torch.randn(rank, self.in_features) * 0.01)
					self.B = nn.Parameter(torch.zeros(self.out_features, rank))
					self.scaling = alpha / rank
				def forward(self, x):
					delta = (self.B @ self.A) * self.scaling
					return F.linear(x, self.weight + delta, self.bias)

			for mod_name, mod in self.timesfm.named_modules():
				if mod is self.timesfm: continue
				if 'g_l' in getattr(mod, "_parameters", {}): continue
				if isinstance(mod, nn.Embedding): continue								# skip embeddings
				if not hasattr(mod, 'in_features') or not hasattr(mod, 'out_features'):
					continue                     										# skip non-linear
				if mod.weight.ndim == 2:
					lora_mod = LoRALinear(mod)
					_replace(self.timesfm, mod_name, lora_mod)
		
		def pad_tensor(self, x):

			B, L = x.shape
			device = x.device
			dtype = x.dtype
			
			if L < self.context_len:
				padded_input = torch.zeros((B, self.context_len), device=device, dtype=dtype)
				padded_input[:, -L:] = x
				padding = torch.ones((B, self.context_len), device=device, dtype=dtype)
				padding[:, -L:] = 0
			else:
				padded_input = x[:, -self.context_len:]
				padding = torch.zeros((B, self.context_len), device=device, dtype=dtype)
			
			freq = torch.zeros((B, 1), device=device, dtype=torch.long)
			
			return padded_input, torch.cat((padding,torch.zeros((B,self.lookahead),device=device,dtype=dtype)),dim=-1), freq
		
		@staticmethod
		def print_mem(s):
			# for debugging
			print(f"{torch.cuda.memory_allocated()/1e6:.2f} MB. INFO: {s}")

		def forward(self, x, weather=None):
			# self.print_mem("Very beginning")
			padded_inp, padding, freq = self.pad_tensor(x)
			if weather is not None:
				if self.weather_model is False:
					# compile should not break graph in this if-condition
					_bad_input("Can't provide weather input with no weather model!")
				else:
					# self.print_mem("before vt_embed")
					weather = self.vt_embed(weather)
					# if self.debug:
					# 	stats = (weather.mean().item(), weather.std().item(), weather.min().item(), weather.max().item())
					# 	print("weather stats:", stats)
					weather = weather + self.adapter_emb(weather).unsqueeze(0).repeat(weather.shape[0],1,1)
					# self.print_mem("after vt_embed, before adapter")
					additional_tokens = []
					for layer in self.adapter.layers:
						weather = layer(weather, src_mask=self.src_mask)
						additional_tokens.append(weather[:,-self.weather_tokens_to_use:,:].clone())
			else:
				additional_tokens = None
			# self.print_mem("after mult, before timesfm")
			out = self.timesfm.decode(padded_inp,padding,freq,self.lookahead,additional_tokens=additional_tokens)[0] # ignoring quantiles
			# self.print_mem("after timesfm")
			return out
		
class TimesFM2(nn.Module):
		
        # TimesFM modification used for using a second timesfm for xreg

		def __init__(self, lookback: int = 512, lookahead: int = 96, context_len: int = 512,
						weather_features = 7,lora_rank = None,
						ckpt='/home/shourya01/timesfm/timesfm-1.0-200m-pytorch/torch_model.ckpt',
					):

			super(TimesFM2, self).__init__()
			
			self.timesfm = PatchedTimeSeriesDecoder(TimesFMConfig())
			self.timesfm2 = PatchedTimeSeriesDecoder(TimesFMConfig())
			self.lookback, self.lookahead = lookback, lookahead
			self.context_len = context_len
			self.ckpt = ckpt

			if lora_rank is not None:
				self._apply_lora_to_timesfm(rank=lora_rank)

			self.proj = nn.Linear(in_features=weather_features,out_features=1)


		def load_weights(self, print_missing_unexpected: bool = False):

			if self.ckpt is not None:
				results = self.timesfm.load_state_dict(torch.load(self.ckpt,map_location="cpu"),strict=False)
				results2 = self.timesfm2.load_state_dict(torch.load(self.ckpt,map_location="cpu"),strict=False)
				if print_missing_unexpected:
					print(f"Results of loading weights into TimesFM1: {results}")
					print(f"Results of loading weights into TimesFM2: {results}")
			else:
				raise ValueError("Checkpoint path not provided.")
			
		def _apply_lora_to_timesfm(self, rank: int = 4, alpha: float = 1.0):
			# …freeze params as before…
			for n, p in self.timesfm.named_parameters():
				p.requires_grad = False
			for n, p in self.timesfm2.named_parameters():
				p.requires_grad = False

			def _replace(root, name, new_mod):
				parts = name.split('.')
				parent = root
				for p in parts[:-1]:
					parent = getattr(parent, p)
				setattr(parent, parts[-1], new_mod)

			class LoRALinear(nn.Module):
				def __init__(self, orig: nn.Linear):
					super().__init__()
					self.in_features = orig.in_features
					self.out_features = orig.out_features
					self.bias = orig.bias
					self.weight = orig.weight
					self.weight.requires_grad = False
					self.A = nn.Parameter(torch.randn(rank, self.in_features) * 0.01)
					self.B = nn.Parameter(torch.zeros(self.out_features, rank))
					self.scaling = alpha / rank
				def forward(self, x):
					delta = (self.B @ self.A) * self.scaling
					return F.linear(x, self.weight + delta, self.bias)

			for mod_name, mod in self.timesfm.named_modules():
				if mod is self.timesfm: continue
				if 'g_l' in getattr(mod, "_parameters", {}): continue
				if isinstance(mod, nn.Embedding): continue								# skip embeddings
				if not hasattr(mod, 'in_features') or not hasattr(mod, 'out_features'):
					continue                     										# skip non-linear
				if mod.weight.ndim == 2:
					lora_mod = LoRALinear(mod)
					_replace(self.timesfm, mod_name, lora_mod)

			for mod_name, mod in self.timesfm2.named_modules():
				if mod is self.timesfm2: continue
				if 'g_l' in getattr(mod, "_parameters", {}): continue
				if isinstance(mod, nn.Embedding): continue								# skip embeddings
				if not hasattr(mod, 'in_features') or not hasattr(mod, 'out_features'):
					continue                     										# skip non-linear
				if mod.weight.ndim == 2:
					lora_mod = LoRALinear(mod)
					_replace(self.timesfm2, mod_name, lora_mod)
		
		def pad_tensor(self, x):

			B, L = x.shape
			device = x.device
			dtype = x.dtype
			
			if L < self.context_len:
				padded_input = torch.zeros((B, self.context_len), device=device, dtype=dtype)
				padded_input[:, -L:] = x
				padding = torch.ones((B, self.context_len), device=device, dtype=dtype)
				padding[:, -L:] = 0
			else:
				padded_input = x[:, -self.context_len:]
				padding = torch.zeros((B, self.context_len), device=device, dtype=dtype)
			
			freq = torch.zeros((B, 1), device=device, dtype=torch.long)
			
			return padded_input, torch.cat((padding,torch.zeros((B,self.lookahead),device=device,dtype=dtype)),dim=-1), freq
		
		def pad_tensor2(self, x): # specifically for batched nputs
			B, L, F = x.shape                      # (batch, seq, features)
			device, dtype = x.device, x.dtype

			if L < self.context_len:               # --- data ---
				padded = torch.zeros((B, self.context_len, F), device=device, dtype=dtype)
				padded[:, -L:, :] = x
				padmask = torch.ones((B, self.context_len), device=device, dtype=dtype)
				padmask[:, -L:] = 0
			else:
				padded  = x[:, -self.context_len:, :]
				padmask = torch.zeros((B, self.context_len), device=device, dtype=dtype)

			# grow mask to (B, ctx+lookahead, F)
			padmask = torch.cat(
				(padmask, torch.zeros((B, self.lookahead), device=device, dtype=dtype)),
				dim=-1
			).unsqueeze(-1).expand(-1, -1, F)

			# freq → (B, 1, F)
			freq = torch.zeros((B, 1, F), device=device, dtype=torch.long)

			return padded, padmask, freq
		
		@staticmethod
		def print_mem(s):
			# for debugging
			print(f"{torch.cuda.memory_allocated()/1e6:.2f} MB. INFO: {s}")

		def merge_BF(self, x: torch.Tensor):
			B, F = x.shape[0], x.shape[-1]               # sizes we’ll need to undo
			x = torch.movedim(x, -1, 1).contiguous()     # (B, F, *mid)
			x = x.reshape(B * F, *x.shape[2:])           # (B·F, *mid)
			return x, (B, F)                             # keep (B, F) for restore

		def unmerge_BF(self, x_flat: torch.Tensor, BF: tuple):
			B, F = BF
			x = x_flat.reshape(B, F, *x_flat.shape[1:])  # (B, F, *mid)
			x = torch.movedim(x, 1, -1)                  # (B, *mid, F)
			return x

		def forward(self, x, weather=None):
			padded_inp, padding, freq = self.pad_tensor(x)
			out = self.timesfm.decode(padded_inp,padding,freq,self.lookahead)[0]
			if weather is not None:
				padded_inp_weather, padding_weather, freq_weather = self.pad_tensor2(weather)
				padded_inp_weather, _bf = self.merge_BF(padded_inp_weather)
				padding_weather, _ = self.merge_BF(padding_weather)
				freq_weather, _ = self.merge_BF(freq_weather)
				out_weather = self.timesfm2.decode(padded_inp_weather, padding_weather, freq_weather, self.lookahead)[0]
				out_weather = self.unmerge_BF(out_weather, _bf)
				return out + self.proj(out_weather).squeeze(-1), out_weather
			else:
				return out
			

class TimesFM3(nn.Module):

		def __init__(self, lookback: int = 512, lookahead: int = 96, context_len: int = 512,
						lora_rank = None,
						weather_model = False,
						weather_tokens_to_use = 6,
						weather_size_reduction = 4,
						weather_decoder_layers = 10,
						ckpt='/home/shourya01/timesfm/timesfm-1.0-200m-pytorch/torch_model.ckpt',
						register_grad_nan_hooks=False):

			super(TimesFM3, self).__init__()
			
			self.timesfm = PatchedTimeSeriesDecoder(TimesFMConfig())
			self.lookback, self.lookahead = lookback, lookahead
			self.context_len = context_len
			self.ckpt = ckpt

			if lora_rank is not None:
				self._apply_lora_to_timesfm(rank=lora_rank)

			self.weather_model = weather_model
			if self.weather_model:
				assert weather_decoder_layers <= self.timesfm.config.num_layers, "Companion transformer cannot have more layers than TimesFM."
				self.weather_tokens_to_use = weather_tokens_to_use
				self.wsr = weather_size_reduction
				self.vt_embed = PointWeatherEmbed(in_channels=5, token_dim=self.timesfm.config.hidden_size)
				self.adapter_emb = SinusoidalEmbedding(d_model=self.timesfm.config.hidden_size, max_len = 1000)
				self.adapter_enc = nn.TransformerEncoderLayer(d_model=self.timesfm.config.hidden_size, 
												  nhead=self.timesfm.config.num_heads,
												  batch_first=True,
												  activation=F.silu)
				self.adapter = nn.TransformerEncoder(self.adapter_enc,num_layers=weather_decoder_layers)
				mask = nn.Transformer.generate_square_subsequent_mask((lookback+lookahead)//4) # weather runs 4 times slower that energy
				self.register_buffer("src_mask",mask)
				
			# # Debugging NaN grads
			# self.debug = register_grad_nan_hooks
			# if self.debug:
			# 	def nan_hook(m, inp, out):
			# 		def chk(t,name):
			# 			if isinstance(t, torch.Tensor) and (torch.isnan(t).any() or torch.isinf(t).any()):
			# 				print(f"{name} ← NaN/Inf in {m.__class__.__name__}")
			# 		for i,x in enumerate(inp):
			# 			chk(x, f"inp#{i}")
			# 		for i,x in enumerate(out if isinstance(out,(list,tuple)) else (out,)):
			# 			chk(x, f"out#{i}")
			# 	for m in self.adapter.modules():
			# 		m.register_forward_hook(nan_hook)

		def load_weights(self, print_missing_unexpected: bool = False):

			if self.ckpt is not None:
				results = self.timesfm.load_state_dict(torch.load(self.ckpt,map_location="cpu"),strict=False)
				if print_missing_unexpected:
					print(f"Results of loading weights: {results}")
			else:
				raise ValueError("Checkpoint path not provided.")
			
		def _apply_lora_to_timesfm(self, rank: int = 4, alpha: float = 1.0):
			# …freeze params as before…
			for n, p in self.timesfm.named_parameters():
				p.requires_grad = (n.split('.')[-1] == 'g_l')

			def _replace(root, name, new_mod):
				parts = name.split('.')
				parent = root
				for p in parts[:-1]:
					parent = getattr(parent, p)
				setattr(parent, parts[-1], new_mod)

			class LoRALinear(nn.Module):
				def __init__(self, orig: nn.Linear):
					super().__init__()
					self.in_features = orig.in_features
					self.out_features = orig.out_features
					self.bias = orig.bias
					self.weight = orig.weight
					self.weight.requires_grad = False
					self.A = nn.Parameter(torch.randn(rank, self.in_features) * 0.01)
					self.B = nn.Parameter(torch.zeros(self.out_features, rank))
					self.scaling = alpha / rank
				def forward(self, x):
					delta = (self.B @ self.A) * self.scaling
					return F.linear(x, self.weight + delta, self.bias)

			for mod_name, mod in self.timesfm.named_modules():
				if mod is self.timesfm: continue
				if 'g_l' in getattr(mod, "_parameters", {}): continue
				if isinstance(mod, nn.Embedding): continue								# skip embeddings
				if not hasattr(mod, 'in_features') or not hasattr(mod, 'out_features'):
					continue                     										# skip non-linear
				if mod.weight.ndim == 2:
					lora_mod = LoRALinear(mod)
					_replace(self.timesfm, mod_name, lora_mod)
		
		def pad_tensor(self, x):

			B, L = x.shape
			device = x.device
			dtype = x.dtype
			
			if L < self.context_len:
				padded_input = torch.zeros((B, self.context_len), device=device, dtype=dtype)
				padded_input[:, -L:] = x
				padding = torch.ones((B, self.context_len), device=device, dtype=dtype)
				padding[:, -L:] = 0
			else:
				padded_input = x[:, -self.context_len:]
				padding = torch.zeros((B, self.context_len), device=device, dtype=dtype)
			
			freq = torch.zeros((B, 1), device=device, dtype=torch.long)
			
			return padded_input, torch.cat((padding,torch.zeros((B,self.lookahead),device=device,dtype=dtype)),dim=-1), freq
		
		@staticmethod
		def print_mem(s):
			# for debugging
			print(f"{torch.cuda.memory_allocated()/1e6:.2f} MB. INFO: {s}")

		def forward(self, x, weather=None):
			# self.print_mem("Very beginning")
			padded_inp, padding, freq = self.pad_tensor(x)
			if weather is not None:
				if self.weather_model is False:
					# compile should not break graph in this if-condition
					_bad_input("Can't provide weather input with no weather model!")
				else:
					# self.print_mem("before vt_embed")
					weather = self.vt_embed(weather)
					# if self.debug:
					# 	stats = (weather.mean().item(), weather.std().item(), weather.min().item(), weather.max().item())
					# 	print("weather stats:", stats)
					weather = weather + self.adapter_emb(weather).unsqueeze(0).repeat(weather.shape[0],1,1)
					# self.print_mem("after vt_embed, before adapter")
					additional_tokens = []
					for layer in self.adapter.layers:
						weather = layer(weather, src_mask=self.src_mask)
						additional_tokens.append(weather[:,-self.weather_tokens_to_use:,:].clone())
			else:
				additional_tokens = None
			# self.print_mem("after mult, before timesfm")
			out = self.timesfm.decode(padded_inp,padding,freq,self.lookahead,additional_tokens=additional_tokens)[0] # ignoring quantiles
			# self.print_mem("after timesfm")
			return out
				