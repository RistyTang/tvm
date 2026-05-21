from tvm.contrib import c6678


def test_generate_c6678_matmul_source_for_ddr():
    mod = c6678.build_matmul_module(
        M=64,
        N=64,
        K=64,
        dtype="float32",
        target="c6678",
        a_scope="ddr",
        b_scope="ddr",
        c_scope="ddr",
        use_multicore=True,
        activation="relu",
        bias_broadcast=True,
    )

    source = mod.inspect_source("c6678")
    assert "fp_matmul_fusion_s" in source
    assert "void fp_matmul_fusion_s(" in source
    assert "void fp_matmul_fusion_p(" not in source
    assert "static void fp_matmul_fusion_s_impl(" in source
    assert "dma_trans_2d" in source
    assert "C6678MatmulConfig" in source


def test_generate_c6678_matmul_source_for_l2():
    mod = c6678.build_matmul_module(
        M=64,
        N=64,
        K=64,
        dtype="float32",
        target="c6678",
        a_scope="l2",
        b_scope="l2",
        c_scope="l2",
        use_multicore=False,
        activation="relu",
        bias_broadcast=True,
    )

    source = mod.inspect_source("c6678")
    assert "fp_matmul_fusion_p" in source
    assert "void fp_matmul_fusion_p(" in source
    assert "void fp_matmul_fusion_s(" not in source
    assert "static void fp_matmul_fusion_p_impl(" in source
