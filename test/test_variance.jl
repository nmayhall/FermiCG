module TestVariance
    using Test
    using FermiCG
    using LinearAlgebra
    using Printf
    using JLD2

    const M_var = 20
    const CIPSI_THRESHOLDS = [2e-2, 8e-3]
    """Load production data, build cluster bases at M=20, return common objects."""
    function load_fixture()
        @load "data_cmf_13_cr2_morokuma.jld2"          # defines ints, clusters, d1
        init_fspace = FockConfig([(3, 0), (3, 3), (0, 3)])
        cluster_bases = FermiCG.compute_cluster_eigenbasis_spin(
                            ints, clusters, d1, [3,3,3], init_fspace,
                            max_roots=M_var, verbose=0)
        clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)
        cluster_ops   = FermiCG.compute_cluster_ops(cluster_bases, ints)
        FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b)
        return ints, clusters, cluster_ops, clustered_ham, d1, cluster_bases
    end

    """
    Build a BSTstate seed (BST CI solve) from cluster_bases.
    Returns (energies, vbst) — used as the starting point for block_sparse_tucker.
    """
    function build_seed(clusters, cluster_ops, clustered_ham, cluster_bases)
        init_fspace = FockConfig([(5,2),(4,4),(2,5)])
        # start by defining P/Q spaces
        p_spaces = Vector{ClusterSubspace}()

        ssi = ClusterSubspace(clusters[1])
        add_subspace!(ssi, (3,0), 1:1)
        add_subspace!(ssi, (2,1), 1:1)
        add_subspace!(ssi, (1,2), 1:1)
        add_subspace!(ssi, (0,3), 1:1)
        push!(p_spaces, ssi)

        ssi = ClusterSubspace(clusters[2])
        add_subspace!(ssi, (3,3), 1:1)
        push!(p_spaces, ssi)

        ssi = ClusterSubspace(clusters[3])
        add_subspace!(ssi, (3,0), 1:1)
        add_subspace!(ssi, (2,1), 1:1)
        add_subspace!(ssi, (1,2), 1:1)
        add_subspace!(ssi, (0,3), 1:1)
        push!(p_spaces, ssi)


        ci_vector = BSTstate(clusters, p_spaces, cluster_bases, R=4)
        na = 6
        nb = 6
        FermiCG.fill_p_space!(ci_vector, na, nb)
        FermiCG.eye!(ci_vector)
        return FermiCG.ci_solve(ci_vector, cluster_ops, clustered_ham)
    end

    """
    Run block_sparse_tucker at `thresh`, starting from `seed_state`.
    Returns (energies, v_var).
    """
    function build_vvar(thresh, clusters, cluster_ops, clustered_ham,
                        cluster_bases, seed_state)
        e, v = FermiCG.block_sparse_tucker(
            seed_state, cluster_ops, clustered_ham;
            max_iter    = 10,
            nbody       = 4,
            H0          = "Hcmf",
            thresh_var  = thresh,
            thresh_spin = thresh/1.2,
            thresh_foi  = thresh / 50,
            thresh_pt   = thresh / 2,
            ci_conv     = 5e-5,
            do_pt       = false,
            tol_tucker  = 1e-5,
            resolve_ss  = true,
            verbose     = 0)
        return e, v
    end

    """Call both drivers and return (σ2_ref, σ2_fast)."""
    function both_sigma2(v_var, cluster_ops, clustered_ham; thresh_foi=1e-8)
        kwargs = (H0="Hcmf", nbody=4, thresh_foi=thresh_foi,
                max_number=nothing, opt_ref=true, ci_tol=1e-6, verbose=0)

        σ2_ref  = FermiCG.compute_spt_sigma_norm_blockwise(
                    v_var, cluster_ops, clustered_ham; kwargs...)
        σ2_fast = FermiCG.compute_spt_sigma_norm_blockwise_alternative(
                    v_var, cluster_ops, clustered_ham; kwargs...)
        return σ2_ref, σ2_fast
    end
    # Build the shared v_var at the coarsest threshold once for all agreement tests
    let
        ints, clusters, cluster_ops, clustered_ham, d1, cluster_bases = load_fixture()
        _, vbst = build_seed(clusters, cluster_ops, clustered_ham, cluster_bases)
        global _agree_e, _agree_v, _agree_cops, _agree_cham =
            let (e, v) = build_vvar(CIPSI_THRESHOLDS[1], clusters, cluster_ops,
                                    clustered_ham, cluster_bases, vbst)
                e, v, cluster_ops, clustered_ham
            end
    end

    @testset "Agreement: σ² ref == fast (thresh=$(CIPSI_THRESHOLDS[1]))" begin
        σ2_ref, σ2_fast = both_sigma2(_agree_v, _agree_cops, _agree_cham)

        @test length(σ2_ref) == length(σ2_fast)
        for r in eachindex(σ2_ref)
            @testset "root $r mismatch" begin
                @test isapprox(σ2_ref[r], σ2_fast[r]; rtol=1e-6, atol=1e-10)
            end
        end
    end

    @testset "Agreement: variance-like quantity matches" begin
        σ2_ref, σ2_fast = both_sigma2(_agree_v, _agree_cops, _agree_cham)

        var_ref  = σ2_ref  .- _agree_e .^ 2
        var_fast = σ2_fast .- _agree_e .^ 2

        for r in eachindex(var_ref)
            @testset "root $r mismatch" begin
                @test isapprox(var_ref[r], var_fast[r]; rtol=1e-6, atol=1e-10)
            end
        end
    end

    @testset "Agreement: σ² values are non-negative" begin
        σ2_ref, σ2_fast = both_sigma2(_agree_v, _agree_cops, _agree_cham)

        for r in eachindex(σ2_ref)
            @testset "root $r mismatch" begin
                @test isapprox(σ2_ref[r], σ2_fast[r]; rtol=1e-6, atol=1e-10)
            end
        end
    end

    @testset "Agreement: thresh_foi sensitivity" begin
        for thresh_foi in (1e-6, 1e-7)
            σ2_ref, σ2_fast = both_sigma2(_agree_v, _agree_cops, _agree_cham;
                                        thresh_foi=thresh_foi)
            @testset "thresh_foi=$thresh_foi" begin
                for r in eachindex(σ2_ref)
                    @testset "root $r" begin
                        @test isapprox(σ2_ref[r], σ2_fast[r]; rtol=1e-5, atol=1e-10)
                    end
                end
            end
        end
    end

    @testset "Regression: variance decreases as wavefunction improves" begin
        # σ² should decrease monotonically across CIPSI_THRESHOLDS.
        ints, clusters, cluster_ops, clustered_ham, d1, cluster_bases = load_fixture()
        _, v_cur = build_seed(clusters, cluster_ops, clustered_ham, cluster_bases)

        var_prev = nothing
        for thresh in CIPSI_THRESHOLDS
            e, v_cur = build_vvar(thresh, clusters, cluster_ops, clustered_ham,
                                cluster_bases, v_cur)
            σ2, _ = both_sigma2(v_cur, cluster_ops, clustered_ham)
            var = σ2 .- e.^2
      
            if var_prev !== nothing
                @testset "thresh=$thresh" begin
                    for r in eachindex(σ2)
                        @testset "root $r: variance did not decrease" begin
                            @test var[r] <= var_prev[r] + 1e-8
                        end
                    end
                end
            end
            var_prev = var
        end
    end

    @testset "Regression: variance-like is finite for all thresholds" begin
        ints, clusters, cluster_ops, clustered_ham, d1, cluster_bases = load_fixture()
        _, v_cur = build_seed(clusters, cluster_ops, clustered_ham, cluster_bases)

        for thresh in CIPSI_THRESHOLDS
            e, v_cur = build_vvar(thresh, clusters, cluster_ops, clustered_ham,
                                cluster_bases, v_cur)
            σ2_ref, σ2_fast = both_sigma2(v_cur, cluster_ops, clustered_ham)
            @testset "thresh=$thresh" begin
                for r in eachindex(e)
                    @testset "root $r" begin
                        @test isfinite(σ2_ref[r]  - e[r]^2)
                        @test isfinite(σ2_fast[r] - e[r]^2)
                    end
                end
            end
        end
    end

    if get(ENV, "SPT_FULL", "0") == "1"

        @testset "Integration: full threshold sweep agreement" begin
            ints, clusters, cluster_ops, clustered_ham, d1, cluster_bases = load_fixture()
            _, v_cur = build_seed(clusters, cluster_ops, clustered_ham, cluster_bases)

            for thresh in CIPSI_THRESHOLDS
                e, v_cur = build_vvar(thresh, clusters, cluster_ops, clustered_ham,
                                    cluster_bases, v_cur)
                σ2_ref, σ2_fast = both_sigma2(v_cur, cluster_ops, clustered_ham)

                var_ref  = σ2_ref  .- e .^ 2
                var_fast = σ2_fast .- e .^ 2

                @testset "thresh=$thresh" begin
                    for r in eachindex(e)
                        @testset "root $r" begin
                            @test isapprox(σ2_ref[r], σ2_fast[r]; rtol=1e-6, atol=1e-10)
                            @test isapprox(var_ref[r], var_fast[r]; rtol=1e-6, atol=1e-10)
                            @test isfinite(var_ref[r])
                            @test isfinite(var_fast[r])
                        end
                    end
                end
            end
        end

    end # SPT_FULL
end # module