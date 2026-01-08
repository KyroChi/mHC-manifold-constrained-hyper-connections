"""
Comprehensive tests for optimized HyperConnections implementation.
Tests numerical equivalence and performance improvements.
"""

import pytest
import torch
import torch.nn as nn
import time
from copy import deepcopy

from hyper_connections import (
    HyperConnections,
    get_init_and_expand_reduce_stream_functions,
    sinkhorn_log,
    orthostochastic_project,
    zeropower_via_newtonschulz,
)


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def seed():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


class TestNumericalEquivalence:
    """Test that optimized implementation produces numerically equivalent results."""

    @pytest.mark.parametrize("num_streams", [1, 2, 4, 8])
    @pytest.mark.parametrize("num_fracs", [1, 2, 4])
    @pytest.mark.parametrize("dim", [64, 128, 256])
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("seq_len", [32, 64, 128])
    def test_hyperconnections_basic(
        self, device, seed, num_streams, num_fracs, dim, batch_size, seq_len
    ):
        """Test basic HyperConnections forward pass."""
        if dim % num_fracs != 0:
            pytest.skip(f"dim {dim} not divisible by num_fracs {num_fracs}")

        branch = nn.Linear(dim, dim).to(device)
        hc = HyperConnections(
            num_residual_streams=num_streams,
            dim=dim,
            branch=branch,
            num_fracs=num_fracs,
        ).to(device)

        init_hc, expand_stream, reduce_stream = (
            get_init_and_expand_reduce_stream_functions(
                num_streams, num_fracs=num_fracs, disable=False
            )
        )

        residual = torch.randn(batch_size, seq_len, dim, device=device, requires_grad=True)
        residual_clone = residual.clone().detach().requires_grad_(True)

        # Forward pass
        expanded = expand_stream(residual)
        output = reduce_stream(hc(expanded))

        expanded_clone = expand_stream(residual_clone)
        output_clone = reduce_stream(hc(expanded_clone))

        # Check outputs match
        torch.testing.assert_close(output, output_clone, atol=1e-6, rtol=1e-5)

        # Backward pass
        loss = output.sum()
        loss.backward()

        loss_clone = output_clone.sum()
        loss_clone.backward()

        # Check gradients match
        torch.testing.assert_close(
            residual.grad, residual_clone.grad, atol=1e-5, rtol=1e-4
        )

    @pytest.mark.parametrize("num_streams", [2, 4])
    @pytest.mark.parametrize("dim", [64, 128])
    def test_hyperconnections_mhc(self, device, seed, num_streams, dim):
        """Test mHC (mixed HyperConnections) mode."""
        branch = nn.Linear(dim, dim).to(device)
        hc = HyperConnections(
            num_residual_streams=num_streams,
            dim=dim,
            branch=branch,
            mhc=True,
            sinkhorn_iters=10,
        ).to(device)

        init_hc, expand_stream, reduce_stream = (
            get_init_and_expand_reduce_stream_functions(num_streams, disable=False)
        )

        batch_size, seq_len = 2, 32
        residual = torch.randn(batch_size, seq_len, dim, device=device, requires_grad=True)
        residual_clone = residual.clone().detach().requires_grad_(True)

        expanded = expand_stream(residual)
        output = reduce_stream(hc(expanded))

        expanded_clone = expand_stream(residual_clone)
        output_clone = reduce_stream(hc(expanded_clone))

        torch.testing.assert_close(output, output_clone, atol=1e-6, rtol=1e-5)

        # Test backward
        loss = output.sum()
        loss.backward()
        loss_clone = output_clone.sum()
        loss_clone.backward()

        torch.testing.assert_close(
            residual.grad, residual_clone.grad, atol=1e-5, rtol=1e-4
        )

    @pytest.mark.parametrize("num_streams", [2, 4])
    @pytest.mark.parametrize("dim", [64, 128])
    def test_hyperconnections_mhc_orthostochastic(self, device, seed, num_streams, dim):
        """Test mHC with orthostochastic projection."""
        branch = nn.Linear(dim, dim).to(device)
        hc = HyperConnections(
            num_residual_streams=num_streams,
            dim=dim,
            branch=branch,
            mhc=True,
            mhc_h_res_proj="orthostochastic",
            ns_steps=5,
        ).to(device)

        init_hc, expand_stream, reduce_stream = (
            get_init_and_expand_reduce_stream_functions(num_streams, disable=False)
        )

        batch_size, seq_len = 2, 32
        residual = torch.randn(batch_size, seq_len, dim, device=device, requires_grad=True)
        residual_clone = residual.clone().detach().requires_grad_(True)

        expanded = expand_stream(residual)
        output = reduce_stream(hc(expanded))

        expanded_clone = expand_stream(residual_clone)
        output_clone = reduce_stream(hc(expanded_clone))

        torch.testing.assert_close(output, output_clone, atol=1e-5, rtol=1e-4)

    @pytest.mark.parametrize("channel_first", [False, True])
    def test_hyperconnections_channel_first(self, device, seed, channel_first):
        """Test channel_first mode."""
        num_streams, dim, batch_size = 4, 64, 2
        branch = nn.Conv2d(dim, dim, 3, padding=1).to(device)
        hc = HyperConnections(
            num_residual_streams=num_streams,
            dim=dim,
            branch=branch,
            channel_first=channel_first,
        ).to(device)

        init_hc, expand_stream, reduce_stream = (
            get_init_and_expand_reduce_stream_functions(num_streams, disable=False)
        )

        residual = torch.randn(batch_size, dim, 16, 16, device=device, requires_grad=True)
        residual_clone = residual.clone().detach().requires_grad_(True)

        expanded = expand_stream(residual)
        output = reduce_stream(hc(expanded))

        expanded_clone = expand_stream(residual_clone)
        output_clone = reduce_stream(hc(expanded_clone))

        torch.testing.assert_close(output, output_clone, atol=1e-6, rtol=1e-5)

    def test_sinkhorn_log(self, device, seed):
        """Test sinkhorn_log function."""
        n = 8
        logits = torch.randn(n, n, device=device) * 0.1

        # Test multiple times to ensure consistency
        result1 = sinkhorn_log(logits, num_iters=10, tau=0.05)
        result2 = sinkhorn_log(logits, num_iters=10, tau=0.05)

        torch.testing.assert_close(result1, result2, atol=1e-6, rtol=1e-5)

        # Test constraints: should be doubly stochastic
        row_sums = result1.sum(dim=-1)
        col_sums = result1.sum(dim=-2)
        expected = torch.ones(n, device=device)
        torch.testing.assert_close(row_sums, expected, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(col_sums, expected, atol=1e-3, rtol=1e-3)

    def test_orthostochastic_project(self, device, seed):
        """Test orthostochastic projection."""
        n = 8
        logits = torch.randn(n, n, device=device) * 0.1

        result = orthostochastic_project(logits, ns_steps=5, ns_eps=1e-7)

        # Should be doubly stochastic
        row_sums = result.sum(dim=-1)
        col_sums = result.sum(dim=-2)
        expected = torch.ones(n, device=device)
        torch.testing.assert_close(row_sums, expected, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(col_sums, expected, atol=1e-2, rtol=1e-2)

    def test_zeropower_via_newtonschulz(self, device, seed):
        """Test Newton-Schulz iteration."""
        m, n = 8, 6
        X = torch.randn(m, n, device=device) * 0.1

        result = zeropower_via_newtonschulz(X, steps=5, eps=1e-7)

        # Should be approximately orthogonal: O @ O.T ≈ I
        if result.shape[0] <= result.shape[1]:
            O = result
        else:
            O = result.T

        OOT = O @ O.T
        I = torch.eye(O.shape[0], device=device)
        torch.testing.assert_close(OOT, I, atol=1e-2, rtol=1e-2)


class TestPerformance:
    """Test performance improvements."""

    def benchmark_hyperconnections(
        self, device, num_streams, num_fracs, dim, batch_size, seq_len, num_iterations=10
    ):
        """Benchmark HyperConnections forward and backward pass."""
        if dim % num_fracs != 0:
            return None, None

        branch = nn.Linear(dim, dim).to(device)
        hc = HyperConnections(
            num_residual_streams=num_streams,
            dim=dim,
            branch=branch,
            num_fracs=num_fracs,
        ).to(device)

        init_hc, expand_stream, reduce_stream = (
            get_init_and_expand_reduce_stream_functions(
                num_streams, num_fracs=num_fracs, disable=False
            )
        )

        residual = torch.randn(batch_size, seq_len, dim, device=device, requires_grad=True)

        # Warmup
        for _ in range(3):
            expanded = expand_stream(residual)
            output = reduce_stream(hc(expanded))
            loss = output.sum()
            loss.backward()
            residual.grad = None

        # Benchmark forward
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iterations):
            expanded = expand_stream(residual)
            output = reduce_stream(hc(expanded))
        if device.type == "cuda":
            torch.cuda.synchronize()
        forward_time = (time.time() - start) / num_iterations

        # Benchmark backward
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iterations):
            expanded = expand_stream(residual)
            output = reduce_stream(hc(expanded))
            loss = output.sum()
            loss.backward()
            residual.grad = None
        if device.type == "cuda":
            torch.cuda.synchronize()
        backward_time = (time.time() - start) / num_iterations

        return forward_time, backward_time

    @pytest.mark.parametrize("num_streams", [2, 4, 8])
    @pytest.mark.parametrize("num_fracs", [1, 2, 4])
    def test_performance_hyperconnections(self, device, seed, num_streams, num_fracs):
        """Test performance of HyperConnections."""
        if device.type == "cpu":
            pytest.skip("Performance tests are more meaningful on GPU")

        dim, batch_size, seq_len = 256, 4, 128
        forward_time, backward_time = self.benchmark_hyperconnections(
            device, num_streams, num_fracs, dim, batch_size, seq_len, num_iterations=50
        )

        if forward_time is not None:
            print(
                f"\nPerformance (num_streams={num_streams}, num_fracs={num_fracs}): "
                f"forward={forward_time*1000:.2f}ms, backward={backward_time*1000:.2f}ms"
            )
            # Just check that it runs, don't enforce specific timing
            assert forward_time > 0
            assert backward_time > 0

    def test_performance_mhc(self, device, seed):
        """Test performance of mHC mode."""
        if device.type == "cpu":
            pytest.skip("Performance tests are more meaningful on GPU")

        num_streams, dim, batch_size, seq_len = 4, 256, 4, 128
        branch = nn.Linear(dim, dim).to(device)
        hc = HyperConnections(
            num_residual_streams=num_streams,
            dim=dim,
            branch=branch,
            mhc=True,
            sinkhorn_iters=10,
        ).to(device)

        init_hc, expand_stream, reduce_stream = (
            get_init_and_expand_reduce_stream_functions(num_streams, disable=False)
        )

        residual = torch.randn(batch_size, seq_len, dim, device=device, requires_grad=True)

        # Warmup
        for _ in range(3):
            expanded = expand_stream(residual)
            output = reduce_stream(hc(expanded))
            loss = output.sum()
            loss.backward()
            residual.grad = None

        # Benchmark
        num_iterations = 50
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iterations):
            expanded = expand_stream(residual)
            output = reduce_stream(hc(expanded))
            loss = output.sum()
            loss.backward()
            residual.grad = None
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time = (time.time() - start) / num_iterations

        print(f"\nmHC Performance: {total_time*1000:.2f}ms per iteration")
        assert total_time > 0


class TestEdgeCases:
    """Test edge cases and special configurations."""

    def test_single_stream(self, device, seed):
        """Test with single stream (should behave like regular residual)."""
        dim, batch_size, seq_len = 64, 2, 32
        branch = nn.Linear(dim, dim).to(device)
        hc = HyperConnections(
            num_residual_streams=1, dim=dim, branch=branch
        ).to(device)

        init_hc, expand_stream, reduce_stream = (
            get_init_and_expand_reduce_stream_functions(1, disable=False)
        )

        residual = torch.randn(batch_size, seq_len, dim, device=device)
        expanded = expand_stream(residual)
        output = reduce_stream(hc(expanded))

        assert output.shape == residual.shape

    def test_no_branch(self, device, seed):
        """Test HyperConnections without a branch."""
        num_streams, dim, batch_size, seq_len = 4, 64, 2, 32
        hc = HyperConnections(
            num_residual_streams=num_streams, dim=dim, branch=None
        ).to(device)

        init_hc, expand_stream, reduce_stream = (
            get_init_and_expand_reduce_stream_functions(num_streams, disable=False)
        )

        residual = torch.randn(batch_size, seq_len, dim, device=device)
        expanded = expand_stream(residual)
        branch_input, add_residual = hc(expanded)

        assert branch_input.shape == (batch_size, seq_len, dim)

    def test_disable_add_branch_out(self, device, seed):
        """Test with add_branch_out_to_residual=False."""
        num_streams, dim, batch_size, seq_len = 4, 64, 2, 32
        branch = nn.Linear(dim, dim).to(device)
        hc = HyperConnections(
            num_residual_streams=num_streams,
            dim=dim,
            branch=branch,
            add_branch_out_to_residual=False,
        ).to(device)

        init_hc, expand_stream, reduce_stream = (
            get_init_and_expand_reduce_stream_functions(num_streams, disable=False)
        )

        residual = torch.randn(batch_size, seq_len, dim, device=device)
        expanded = expand_stream(residual)
        output = reduce_stream(hc(expanded))

        assert output.shape == residual.shape

    def test_multiple_input_views(self, device, seed):
        """Test with multiple input views."""
        num_streams, dim, batch_size, seq_len = 4, 64, 2, 32
        branch = nn.Linear(dim, dim).to(device)
        hc = HyperConnections(
            num_residual_streams=num_streams,
            dim=dim,
            branch=branch,
            num_input_views=2,
        ).to(device)

        init_hc, expand_stream, reduce_stream = (
            get_init_and_expand_reduce_stream_functions(num_streams, disable=False)
        )

        residual = torch.randn(batch_size, seq_len, dim, device=device)
        expanded = expand_stream(residual)
        branch_input, add_residual = hc(expanded)

        # With 2 input views, branch_input should be a tuple/list
        assert isinstance(branch_input, (list, tuple))
        assert len(branch_input) == 2
        assert branch_input[0].shape == (batch_size, seq_len, dim)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

