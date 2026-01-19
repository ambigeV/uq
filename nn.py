"""
Graph encoder components for DMPNN (Directed Message Passing Neural Network).

This module contains all graph-related components including:
- MolGraph and BatchMolGraph data structures
- BondMessagePassing and AtomMessagePassing modules
- Aggregation modules (Mean, Sum)
- DMPNNEncoder and factory function
- UnifiedTorchModel (custom TorchModel for graph and vector data)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Sequence, Optional, List, Tuple, Iterable, Union
import deepchem as dc
from deepchem.models import TorchModel
from deepchem.data import Dataset
from deepchem.feat import GraphData


# ============================================================
# Graph Data Structures
# ============================================================

@dataclass
class MolGraph:
    """A MolGraph represents the graph featurization of a molecule."""
    V: np.ndarray  # shape: V x d_v (atom features)
    E: np.ndarray  # shape: E x d_e (bond features)
    edge_index: np.ndarray  # shape: 2 x E (edge connectivity)
    rev_edge_index: np.ndarray  # shape: E (reverse edge mapping)


class BatchMolGraph:
    """A BatchMolGraph represents a batch of individual MolGraphs."""
    
    def __init__(self, V, E, edge_index, rev_edge_index, batch):
        self.V = V  # shape: num_nodes x d_v (atom features)
        self.E = E  # shape: num_edges x d_e (bond features)
        self.edge_index = edge_index  # shape: 2 x num_edges (edge connectivity)
        self.rev_edge_index = rev_edge_index  # shape: num_edges (reverse edge mapping)
        self.batch = batch  # shape: num_nodes (batch indices)
    
    @classmethod
    def from_molgraphs(cls, mgs: Sequence[MolGraph]) -> 'BatchMolGraph':
        """Create a BatchMolGraph from a sequence of MolGraph objects."""
        Vs = []
        Es = []
        edge_indexes = []
        rev_edge_indexes = []
        batch_indexes = []
        
        num_nodes = 0
        num_edges = 0
        for i, mg in enumerate(mgs):
            Vs.append(mg.V)
            Es.append(mg.E)
            edge_indexes.append(mg.edge_index + num_nodes)
            rev_edge_indexes.append(mg.rev_edge_index + num_edges)
            batch_indexes.append([i] * len(mg.V))
            
            num_nodes += mg.V.shape[0]
            num_edges += mg.edge_index.shape[1]
        
        return cls(
            V=torch.from_numpy(np.concatenate(Vs)).float(),
            E=torch.from_numpy(np.concatenate(Es)).float(),
            edge_index=torch.from_numpy(np.hstack(edge_indexes)).long(),
            rev_edge_index=torch.from_numpy(np.concatenate(rev_edge_indexes)).long(),
            batch=torch.tensor(np.concatenate(batch_indexes)).long()
        )
    
    def to(self, device: str | torch.device):
        """Move all tensors to the specified device."""
        self.V = self.V.to(device)
        self.E = self.E.to(device)
        self.edge_index = self.edge_index.to(device)
        self.rev_edge_index = self.rev_edge_index.to(device)
        self.batch = self.batch.to(device)
        return self


def graphdata_to_batchmolgraph(graph_data_list) -> BatchMolGraph:
    """
    Convert a list of DeepChem GraphData objects to BatchMolGraph.
    
    Args:
        graph_data_list: List of GraphData objects from DeepChem featurizer
    
    Returns:
        BatchMolGraph object
    """
    mgs = []
    for graph_data in graph_data_list:
        # GraphData has: node_features, edge_index, edge_features
        # We need to create rev_edge_index
        edge_index = graph_data.edge_index  # shape: 2 x E
        
        # Create reverse edge index
        # For each edge (src, dst), find the reverse edge (dst, src)
        num_edges = edge_index.shape[1]
        
        # Handle edge case: graph with no edges
        if num_edges == 0:
            rev_edge_index = np.array([], dtype=np.int64)
        else:
            rev_edge_index = np.zeros(num_edges, dtype=np.int64)
            
            # VECTORIZED: Build reverse edge mapping using numpy operations
            # For edge i: (src_i, dst_i), find edge j where (dst_i, src_i) = (src_j, dst_j)
            src_nodes = edge_index[0, :]  # (num_edges,)
            dst_nodes = edge_index[1, :]  # (num_edges,)
            
            # Create arrays for comparison: for each edge, find matching reverse
            # We compare: (dst_i, src_i) with all (src_j, dst_j)
            # Using broadcasting: (num_edges, 1) vs (1, num_edges)
            # For edge i: we want to find edge j where src_j == dst_i AND dst_j == src_i
            src_match = (src_nodes[:, None] == dst_nodes[None, :])  # (num_edges, num_edges) - src_i == dst_j
            dst_match = (dst_nodes[:, None] == src_nodes[None, :])  # (num_edges, num_edges) - dst_i == src_j
            reverse_mask = src_match & dst_match  # (num_edges, num_edges) - True where reverse edge exists
            
            # For each edge, find its reverse edge index
            # argmax returns first True index, or 0 if all False
            reverse_indices = np.argmax(reverse_mask, axis=1)  # (num_edges,) - index of reverse edge
            # Check if reverse actually exists (argmax might return 0 even if no match)
            has_reverse = np.any(reverse_mask, axis=1)  # (num_edges,) - True if reverse exists
            # Use reverse index if exists, otherwise use self (for self-loops or unmatched edges)
            rev_edge_index = np.where(has_reverse, reverse_indices, np.arange(num_edges))
        
        # Create MolGraph
        mg = MolGraph(
            V=graph_data.node_features,  # node_features -> V
            E=graph_data.edge_features if graph_data.edge_features is not None else np.zeros((num_edges, 14)),  # edge_features -> E, default d_e=14
            edge_index=edge_index,
            rev_edge_index=rev_edge_index
        )
        mgs.append(mg)
    
    return BatchMolGraph.from_molgraphs(mgs)


# ============================================================
# Message Passing Modules
# ============================================================

class BondMessagePassing(nn.Module):
    """
    Bond-based Message Passing (DMPNN style).
    
    Implements the exact formulation from chemprop:
    h_{vw}^{(0)} = τ(W_i([x_v || e_{vw}]))
    m_{vw}^{(t)} = Σ_{u∈N(v)\\w} h_{uv}^{(t-1)}
    h_{vw}^{(t)} = τ(h_{vw}^{(0)} + W_h(m_{vw}^{(t)}))
    m_v^{(T)} = Σ_{w∈N(v)} h_{vw}^{(T-1)}
    h_v^{(T)} = τ(W_o([x_v || m_v^{(T)}]))
    """
    
    def __init__(
        self,
        d_v: int = 133,  # Atom feature dimension
        d_e: int = 14,   # Bond feature dimension
        d_h: int = 200,  # Hidden dimension
        depth: int = 3,  # Number of message passing iterations
        dropout: float = 0.0,
        activation: str = "relu",
        bias: bool = True,  # Use bias (as requested)
    ):
        super().__init__()
        self.d_v = d_v
        self.d_e = d_e
        self.d_h = d_h
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        if activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Weight matrices (with bias as requested)
        # W_i: (d_v + d_e) -> d_h
        self.W_i = nn.Linear(d_v + d_e, d_h, bias=bias)
        # W_h: d_h -> d_h
        self.W_h = nn.Linear(d_h, d_h, bias=bias)
        # W_o: (d_v + d_h) -> d_h
        self.W_o = nn.Linear(d_v + d_h, d_h, bias=bias)
    
    @property
    def output_dim(self) -> int:
        return self.d_h
    
    def forward(self, bmg: BatchMolGraph, V_d: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through bond message passing.
        
        Args:
            bmg: BatchMolGraph object
            V_d: Optional vertex descriptors (not used in basic DMPNN)
        
        Returns:
            Node embeddings H_v of shape (num_nodes, d_h)
        """
        V = bmg.V  # (num_nodes, d_v)
        E = bmg.E  # (num_edges, d_e)
        edge_index = bmg.edge_index  # (2, num_edges)
        rev_edge_index = bmg.rev_edge_index  # (num_edges,)
        
        num_nodes = V.shape[0]
        num_edges = E.shape[0]
        
        # Initialize: h_{vw}^{(0)} = τ(W_i([x_v || e_{vw}]))
        # For each edge (v, w), concatenate atom features of v with bond features
        src_nodes = edge_index[0]  # Source nodes for each edge
        V_edge = V[src_nodes]  # (num_edges, d_v) - atom features for source nodes
        edge_input = torch.cat([V_edge, E], dim=1)  # (num_edges, d_v + d_e)
        H_0 = self.activation(self.W_i(edge_input))  # (num_edges, d_h)
        H_0 = self.dropout(H_0)
        
        # Message passing iterations
        H = H_0
        for _ in range(1, self.depth):
            # m_{vw}^{(t)} = Σ_{u∈N(v)\\w} h_{uv}^{(t-1)}
            # For each edge (v, w), sum messages from neighbors of v (excluding w)
            # VECTORIZED VERSION: Use scatter operations instead of Python loop
            
            # Step 1: For each node, aggregate all incoming messages
            # Use scatter_add to sum all incoming edge messages per node
            # dst_nodes = edge_index[1]  # Destination nodes (where edges end)
            dst_nodes = edge_index[1]  # (num_edges,)
            node_messages = torch.zeros(num_nodes, self.d_h, dtype=H.dtype, device=H.device)
            node_messages.scatter_add_(0, dst_nodes.unsqueeze(1).expand(-1, self.d_h), H)
            
            # Step 2: For each edge (v, w), get aggregated messages at node v
            # Then subtract the message from the reverse edge (w, v) if it exists
            src_nodes = edge_index[0]  # Source nodes (where edges start)
            M = node_messages[src_nodes]  # (num_edges, d_h) - messages aggregated at source nodes
            
            # Step 3: Subtract reverse edge messages to exclude them
            # For edge (v, w), we want messages at v excluding the message from (w, v)
            # The reverse edge index is stored in rev_edge_index
            # Only subtract if reverse edge exists (rev_edge_index[i] != i means reverse exists)
            edge_indices = torch.arange(num_edges, device=rev_edge_index.device, dtype=rev_edge_index.dtype)
            reverse_edge_mask = (rev_edge_index != edge_indices)  # True where reverse exists
            reverse_messages = H[rev_edge_index]  # (num_edges, d_h) - messages from reverse edges
            # Only subtract where reverse exists, otherwise subtract 0
            M = M - reverse_messages * reverse_edge_mask.unsqueeze(1).float()
            
            # h_{vw}^{(t)} = τ(h_{vw}^{(0)} + W_h(m_{vw}^{(t)}))
            H = self.activation(H_0 + self.W_h(M))
            H = self.dropout(H)
        
        # Final aggregation: m_v^{(T)} = Σ_{w∈N(v)} h_{vw}^{(T-1)}
        # Aggregate messages for each node using scatter_add
        M_v = torch.zeros(num_nodes, self.d_h, dtype=H.dtype, device=H.device)
        # Use scatter_add: for each node, sum all outgoing edge messages
        src_nodes = edge_index[0]  # Source nodes for each edge
        M_v.scatter_add_(0, src_nodes.unsqueeze(1).expand(-1, self.d_h), H)
        
        # Final output: h_v^{(T)} = τ(W_o([x_v || m_v^{(T)}]))
        node_output = torch.cat([V, M_v], dim=1)  # (num_nodes, d_v + d_h)
        H_v = self.activation(self.W_o(node_output))
        H_v = self.dropout(H_v)
        
        return H_v


class AtomMessagePassing(nn.Module):
    """
    Atom-based Message Passing (for future use).
    
    Implements atom message passing formulation from chemprop.
    """
    
    def __init__(
        self,
        d_v: int = 133,
        d_e: int = 14,
        d_h: int = 300,
        depth: int = 3,
        dropout: float = 0.0,
        activation: str = "relu",
        bias: bool = True,
    ):
        super().__init__()
        self.d_v = d_v
        self.d_e = d_e
        self.d_h = d_h
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        
        if activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # W_i: d_v -> d_h
        self.W_i = nn.Linear(d_v, d_h, bias=bias)
        # W_h: (d_e + d_h) -> d_h
        self.W_h = nn.Linear(d_e + d_h, d_h, bias=bias)
        # W_o: (d_v + d_h) -> d_h
        self.W_o = nn.Linear(d_v + d_h, d_h, bias=bias)
    
    @property
    def output_dim(self) -> int:
        return self.d_h
    
    def forward(self, bmg: BatchMolGraph, V_d: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through atom message passing."""
        V = bmg.V  # (num_nodes, d_v)
        E = bmg.E  # (num_edges, d_e)
        edge_index = bmg.edge_index  # (2, num_edges)
        
        num_nodes = V.shape[0]
        num_edges = E.shape[0]
        
        # Initialize: h_v^{(0)} = τ(W_i(x_v))
        H_0 = self.activation(self.W_i(V))  # (num_nodes, d_h)
        H_0 = self.dropout(H_0)
        
        # Message passing iterations
        H = H_0
        for _ in range(1, self.depth):
            # m_v^{(t)} = Σ_{u∈N(v)} [h_u^{(t-1)} || e_{uv}]
            M = torch.zeros(num_nodes, self.d_h, dtype=H.dtype, device=H.device)
            
            # Use scatter_add for efficient aggregation
            dst_nodes = edge_index[1]  # Destination nodes (where messages are sent)
            src_nodes = edge_index[0]   # Source nodes (where messages come from)
            
            # For each edge, compute message: W_h([h_u || e_{uv}])
            neighbor_h = H[src_nodes]  # (num_edges, d_h)
            neighbor_e = E  # (num_edges, d_e)
            neighbor_msg = torch.cat([neighbor_h, neighbor_e], dim=1)  # (num_edges, d_h + d_e)
            edge_messages = self.W_h(neighbor_msg)  # (num_edges, d_h)
            
            # Aggregate messages at destination nodes
            M.scatter_add_(0, dst_nodes.unsqueeze(1).expand(-1, self.d_h), edge_messages)
            
            # h_v^{(t)} = τ(h_v^{(0)} + m_v^{(t)})
            H = self.activation(H_0 + M)
            H = self.dropout(H)
        
        # Final: h_v^{(T)} = τ(W_o([x_v || m_v^{(T)}]))
        # m_v^{(T)} = Σ_{w∈N(v)} h_w^{(T-1)}
        M_v = torch.zeros(num_nodes, self.d_h, dtype=H.dtype, device=H.device)
        # Aggregate using scatter_add
        dst_nodes = edge_index[1]
        src_nodes = edge_index[0]
        M_v.scatter_add_(0, dst_nodes.unsqueeze(1).expand(-1, self.d_h), H[src_nodes])
        
        node_output = torch.cat([V, M_v], dim=1)  # (num_nodes, d_v + d_h)
        H_v = self.activation(self.W_o(node_output))
        H_v = self.dropout(H_v)
        
        return H_v


# ============================================================
# Aggregation Modules
# ============================================================

class MeanAggregation(nn.Module):
    """Mean aggregation: average node embeddings per graph."""
    
    def forward(self, H_v: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H_v: Node embeddings (num_nodes, d_h)
            batch: Batch indices (num_nodes,)
        
        Returns:
            Graph embeddings (batch_size, d_h)
        """
        batch_size = batch.max().item() + 1
        d_h = H_v.shape[1]
        
        # Use scatter operations for efficiency
        graph_embeddings = torch.zeros(batch_size, d_h, dtype=H_v.dtype, device=H_v.device)
        graph_embeddings.scatter_add_(0, batch.unsqueeze(1).expand(-1, d_h), H_v)
        
        # Compute counts per graph for mean
        counts = torch.zeros(batch_size, dtype=torch.long, device=batch.device)
        counts.scatter_add_(0, batch, torch.ones_like(batch, dtype=torch.long))
        counts = counts.unsqueeze(1).float().clamp(min=1.0)  # Avoid division by zero
        
        graph_embeddings = graph_embeddings / counts
        
        return graph_embeddings


class SumAggregation(nn.Module):
    """Sum aggregation: sum node embeddings per graph."""
    
    def forward(self, H_v: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H_v: Node embeddings (num_nodes, d_h)
            batch: Batch indices (num_nodes,)
        
        Returns:
            Graph embeddings (batch_size, d_h)
        """
        batch_size = batch.max().item() + 1
        d_h = H_v.shape[1]
        
        # Use scatter_add for efficiency
        graph_embeddings = torch.zeros(batch_size, d_h, dtype=H_v.dtype, device=H_v.device)
        graph_embeddings.scatter_add_(0, batch.unsqueeze(1).expand(-1, d_h), H_v)
        
        return graph_embeddings


# ============================================================
# DMPNN Encoder
# ============================================================

class DMPNNEncoder(nn.Module):
    """
    DMPNN encoder: Message Passing + Aggregation.
    Extracts graph-level embeddings from molecular graphs.
    """
    
    def __init__(
        self,
        message_passing: nn.Module,  # BondMessagePassing or AtomMessagePassing
        agg: nn.Module,  # MeanAggregation or SumAggregation
        batch_norm: bool = False,
        hidden_dim: int = 300,
    ):
        super().__init__()
        self.message_passing = message_passing
        self.agg = agg
        self.bn = nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity()
        self.output_dim = hidden_dim
    
    def forward(
        self,
        bmg: BatchMolGraph,
        V_d: Optional[torch.Tensor] = None,
        X_d: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through DMPNN encoder.
        
        Args:
            bmg: BatchMolGraph object
            V_d: Optional vertex descriptors
            X_d: Optional additional graph-level descriptors (for future pre-trained features)
        
        Returns:
            Graph embeddings (batch_size, hidden_dim)
        """
        # Message passing: nodes -> node embeddings
        H_v = self.message_passing(bmg, V_d)
        
        # Aggregation: node embeddings -> graph embeddings
        H = self.agg(H_v, bmg.batch)
        
        # Batch normalization
        H = self.bn(H)
        
        # Optional: concatenate additional descriptors
        if X_d is not None:
            H = torch.cat([H, X_d], dim=1)
            # Note: This changes output_dim, but we'll handle this in create_ffn
        
        return H


def create_dmpnn_encoder(
    d_v: int = 133,
    d_e: int = 14,
    d_h: int = 300,
    depth: int = 3,
    aggregation: str = "mean",
    batch_norm: bool = False,
    message_passing_type: str = "bond",
    dropout: float = 0.0,
    activation: str = "relu",
    bias: bool = True,
    **kwargs
) -> DMPNNEncoder:
    """
    Factory function to create a DMPNN encoder.
    
    Args:
        d_v: Atom feature dimension (default: 133)
        d_e: Bond feature dimension (default: 14)
        d_h: Hidden dimension (default: 300)
        depth: Number of message passing iterations (default: 3)
        aggregation: "mean" or "sum" (default: "mean")
        batch_norm: Whether to apply batch normalization (default: False)
        message_passing_type: "bond" (DMPNN) or "atom" (default: "bond")
        dropout: Dropout rate (default: 0.0)
        activation: Activation function (default: "relu")
        bias: Whether to use bias in linear layers (default: True)
        **kwargs: Additional arguments (ignored)
    
    Returns:
        DMPNNEncoder instance
    """
    # Create message passing module
    if message_passing_type == "bond":
        message_passing = BondMessagePassing(
            d_v=d_v,
            d_e=d_e,
            d_h=d_h,
            depth=depth,
            dropout=dropout,
            activation=activation,
            bias=bias,
        )
    elif message_passing_type == "atom":
        message_passing = AtomMessagePassing(
            d_v=d_v,
            d_e=d_e,
            d_h=d_h,
            depth=depth,
            dropout=dropout,
            activation=activation,
            bias=bias,
        )
    else:
        raise ValueError(f"Unknown message_passing_type: {message_passing_type}")
    
    # Create aggregation module
    if aggregation == "mean":
        agg = MeanAggregation()
    elif aggregation == "sum":
        agg = SumAggregation()
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    
    return DMPNNEncoder(
        message_passing=message_passing,
        agg=agg,
        batch_norm=batch_norm,
        hidden_dim=d_h,
    )


# ============================================================
# Unified TorchModel for Graph and Vector Data
# ============================================================

class UnifiedTorchModel(TorchModel):
    """
    Custom TorchModel that handles both graph data (GraphData → BatchMolGraph) 
    and vector data (numpy arrays → tensors).
    
    This class overrides default_generator() and _prepare_batch() to seamlessly
    convert GraphData objects to BatchMolGraph when encoder_type="dmpnn",
    while maintaining default behavior for vector data (encoder_type="identity").
    """
    
    def __init__(self, model, encoder_type: str = "identity", **kwargs):
        """
        Initialize UnifiedTorchModel.
        
        Args:
            model: PyTorch model (e.g., UnifiedModel)
            encoder_type: "identity" for vector data, "dmpnn" for graph data
            **kwargs: Additional arguments passed to TorchModel
        """
        if encoder_type not in ["identity", "dmpnn"]:
            raise ValueError(f"encoder_type must be 'identity' or 'dmpnn', got '{encoder_type}'")
        
        self.encoder_type = encoder_type
        super(UnifiedTorchModel, self).__init__(model, **kwargs)
    
    def default_generator(
        self,
        dataset: Dataset,
        epochs: int = 1,
        mode: str = 'fit',
        deterministic: bool = True,
        pad_batches: bool = False,
        **kwargs
    ) -> Iterable[Tuple[List, List, List]]:
        """
        Create a generator that iterates batches for a dataset.
        
        For encoder_type="dmpnn": Converts GraphData objects to BatchMolGraph per batch.
        For encoder_type="identity": Uses parent's default generator (vector data).
        
        Args:
            dataset: The data to iterate
            epochs: Number of times to iterate over the full dataset
            mode: 'fit', 'predict', or 'uncertainty'
            deterministic: Whether to iterate in order or shuffle
            pad_batches: Whether to pad each batch to preferred batch size
            
        Returns:
            Generator yielding tuples of (inputs, outputs, weights)
        """
        if self.encoder_type == "identity":
            # Use parent's default generator for vector data
            return super(UnifiedTorchModel, self).default_generator(
                dataset, epochs, mode, deterministic, pad_batches, **kwargs
            )
        
        # encoder_type == "dmpnn": Handle graph data
        for epoch in range(epochs):
            for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
                batch_size=self.batch_size,
                deterministic=deterministic,
                pad_batches=pad_batches
            ):
                # DeepChem's iterbatches returns X_b as numpy array (object dtype) for GraphData
                # Convert to list for easier handling
                if isinstance(X_b, np.ndarray):
                    # Handle numpy array of GraphData objects (object dtype)
                    if X_b.dtype == np.object_ or X_b.dtype == object:
                        X_b = X_b.tolist()  # Convert to Python list
                    else:
                        raise ValueError(
                            f"For encoder_type='dmpnn', expected numpy array with object dtype "
                            f"(containing GraphData), got dtype={X_b.dtype}. "
                            f"Make sure use_graph=True in load_dataset()."
                        )
                elif not isinstance(X_b, (list, tuple)):
                    raise ValueError(
                        f"For encoder_type='dmpnn', expected list/tuple or numpy array of GraphData objects, "
                        f"got {type(X_b)}"
                    )
                
                if len(X_b) == 0:
                    raise ValueError("Empty batch received for encoder_type='dmpnn'")
                
                # Validate that we have GraphData objects
                first_item = X_b[0]
                if not isinstance(first_item, GraphData):
                    raise ValueError(
                        f"For encoder_type='dmpnn', expected GraphData objects, "
                        f"got {type(first_item)}. Make sure use_graph=True in load_dataset()."
                    )
                
                # Convert GraphData list to BatchMolGraph
                batch_molgraph = graphdata_to_batchmolgraph(X_b)
                
                # Yield as list (will be handled in _prepare_batch)
                yield ([batch_molgraph], [y_b], [w_b])
    
    def _prepare_batch(
        self, batch: Tuple[List, List, List]
    ) -> Tuple[Union[BatchMolGraph, torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Prepare batch for model forward pass.
        
        For encoder_type="dmpnn": Handles BatchMolGraph objects and moves to device.
        For encoder_type="identity": Uses parent's _prepare_batch (tensor conversion).
        
        Args:
            batch: Tuple of (inputs, outputs, weights) from default_generator
            
        Returns:
            Tuple of (prepared_inputs, labels, weights) ready for model forward
        """
        inputs, labels, weights = batch
        
        if self.encoder_type == "identity":
            # Use parent's _prepare_batch for vector data
            return super(UnifiedTorchModel, self)._prepare_batch(batch)
        
        # encoder_type == "dmpnn": Handle BatchMolGraph
        if len(inputs) != 1:
            raise ValueError(
                f"For encoder_type='dmpnn', expected single BatchMolGraph in batch, "
                f"got {len(inputs)} items"
            )
        
        batch_molgraph = inputs[0]
        if not isinstance(batch_molgraph, BatchMolGraph):
            raise ValueError(
                f"For encoder_type='dmpnn', expected BatchMolGraph, "
                f"got {type(batch_molgraph)}"
            )
        
        # Move BatchMolGraph to device
        batch_molgraph = batch_molgraph.to(self.device)
        
        # Prepare labels and weights using parent method
        _, labels_tensors, weights_tensors = super(UnifiedTorchModel, self)._prepare_batch(
            ([], labels, weights)
        )
        
        # Return BatchMolGraph (not wrapped in list, as model.forward expects it directly)
        return batch_molgraph, labels_tensors, weights_tensors
    
    def _compute_model_outputs(self, outputs, output_types):
        """
        Override to ensure MC dropout classification outputs are passed correctly.
        
        For MC dropout classification, the model returns a single tensor (B, 2*n_tasks),
        but DeepChem's default behavior with output_types=['prediction'] might extract
        only the first n_tasks elements. We override to pass the full tensor.
        """
        # For MC dropout classification, we need to pass the full output tensor
        # Check if model is UnifiedModel with mc_dropout and classification
        if hasattr(self.model, 'model_type') and hasattr(self.model, 'classification'):
            if self.model.model_type == "mc_dropout" and self.model.classification:
                # For MC dropout classification, return the full output as-is
                # Don't let DeepChem extract/reshape it
                if isinstance(outputs, torch.Tensor):
                    # If it's a single tensor, wrap it in a list to match expected format
                    # but ensure DeepChem doesn't extract from it
                    return [outputs]
        
        # For all other cases, use parent's default behavior
        return super(UnifiedTorchModel, self)._compute_model_outputs(outputs, output_types)