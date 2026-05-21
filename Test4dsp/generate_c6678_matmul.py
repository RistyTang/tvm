import os

from tvm.contrib import c6678


def main():
    mod = c6678.build_matmul_module(
        M=256,
        N=256,
        K=256,
        dtype="float32",
        target="c6678",
        a_scope="ddr",
        b_scope="ddr",
        c_scope="ddr",
        use_multicore=True,
        activation="none",
        bias_broadcast=True,
    )

    source = mod.inspect_source("c6678")
    output_dir = os.path.join(os.path.dirname(__file__), "tests")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "generated_c6678_matmul.c")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(source)

    print(f"Generated C6678 matmul source written to {output_file}")


if __name__ == "__main__":
    main()
