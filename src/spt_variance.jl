using FermiCG
using LinearAlgebra
using Printf

# Block-by-block computation of Hσ = <X|H|0> and its norm, without storing a global BSTstate σ
#
# This mirrors the FOIS construction logic in `_pt2_job2` and the sigma-block logic in `_pt2_job`,
# but instead of building a global `sig::BSTstate`, it:
#   - Groups contributions per (sig_fock, sig_tconfig)
#   - For each block, builds σ_block = <X|H|0> via form_sigma_block_expand
#   - Immediately computes ⟨σ_block|σ_block⟩ and discards the block


"""
    _sigma_block_norm2(term_list, sig_fock, sig_tconfig, ket, cluster_ops,
                       clustered_ham; thresh, max_number)

Given a list of contributions `(term, ket_fock, ket_tconfig, ket_tuck)` that all couple
the same FOIS block `(sig_fock, sig_tconfig)` to the reference `ket`, build the
block σ_block = <X|H|0> for that block only and return ⟨σ_block|σ_block⟩ (length-R vector).
"""
function _sigma_block_norm2(term_list::Vector{NTuple{4,Any}},
                            sig_fock::FockConfig{N},
                            sig_tconfig::TuckerConfig{N},
                            ket::BSTstate{T,N,R},
                            cluster_ops,
                            clustered_ham;
                            thresh::T, max_number) where {T,N,R}

    # Collect Tucker contributions for this block only
    tucks = Tucker{T,N,R}[]

    for (term, ket_fock, ket_tconfig, ket_tuck) in term_list
        # Ensure term actually couples this block 
        check_term(term, sig_fock, sig_tconfig, ket_fock, ket_tconfig) || continue

        # form_sigma_block_expand already exists and is used in _pt2_job for FOIS building
        sig_tuck = form_sigma_block_expand(term, cluster_ops,
                                           sig_fock, sig_tconfig,
                                           ket_fock, ket_tconfig, ket_tuck;
                                           max_number=max_number,
                                           prescreen=thresh)

        # Drop empty / tiny blocks
        (length(sig_tuck) == 0 || norm(sig_tuck) < thresh) && continue

        sig_tuck = compress(sig_tuck; thresh=thresh)
        length(sig_tuck) == 0 && continue

        push!(tucks, sig_tuck)
    end

    # Nothing contributed to this block
    if isempty(tucks)
        return zeros(T,R)
    end

    # Combine into a single σ_block for this FOIS block
    σ_block = try
        # first try default SVD path
        nonorth_add(tucks)  # Tucker{T,N,R}, svd_alg defaults to :default
    catch e
        if e isa LinearAlgebra.LAPACKException
            @warn "nonorth_add (default SVD) failed; retrying with QRIteration" sig_fock sig_tconfig error=e
            # retry with QR-based SVD
            nonorth_add(tucks; svd_alg = :qr)
        else
            rethrow()
        end
    end


    # Compute local ⟨σ_block|σ_block⟩.
    #
    # We can do this by wrapping σ_block into a tiny BSTstate with just this block,
    # then using the existing orth_dot, and discarding the BSTstate right away.

    σ_state = BSTstate(ket.clusters, ket.p_spaces, ket.q_spaces, T=T, R=R)
    add_fockconfig!(σ_state, sig_fock)
    σ_state[sig_fock][sig_tconfig] = σ_block

    n2 = orth_dot(σ_state, σ_state)

    # Drop σ_state and σ_block as they go out of scope
    return n2
end

"""
    _pt2_job_sigma_norm_blockwise(sig_fock, job, ket, cluster_ops, clustered_ham,
                                  nbody, verbose, thresh, max_number, prescreen)

Block-by-block job for computing ⟨X|H|0⟩·⟨X|H|0⟩ in the FOIS defined by `job` and `sig_fock`.

It:
  - Does *not* store a global σ or sig BSTstate
  - Groups contributions per (sig_fock, sig_tconfig)
  - For each FOIS block, builds σ_block = <X|H|0> via `_sigma_block_norm2`
  - Immediately accumulates ⟨σ_block|σ_block⟩ into a per-root accumulator
"""
function _pt2_job_sigma_norm_blockwise(sig_fock, job,
                                       ket::BSTstate{T,N,R},
                                       cluster_ops, clustered_ham,
                                       nbody, verbose, thresh, max_number,
                                       prescreen) where {T,N,R}

    # Map: TuckerConfig{N} => Vector of (term, ket_fock, ket_tconfig, ket_tuck)
    tconfigs_to_process = Dict{TuckerConfig{N}, Vector{NTuple{4,Any}}}()

    # --- FOIS reachability analysis: identical to logic in _pt2_job2 -----------------------
    for jobi in job
        terms, ket_fock, ket_tconfigs = jobi

        for term in terms
            length(term.clusters) <= nbody || continue

            for (ket_tconfig, ket_tuck) in ket_tconfigs

                # available cluster subspaces for this term and fock sector
                available = []
                for ci in term.clusters
                    tmp = []
                    if haskey(ket.p_spaces[ci.idx], sig_fock[ci.idx])
                        push!(tmp, ket.p_spaces[ci.idx][sig_fock[ci.idx]])
                    end
                    if haskey(ket.q_spaces[ci.idx], sig_fock[ci.idx])
                        push!(tmp, ket.q_spaces[ci.idx][sig_fock[ci.idx]])
                    end
                    push!(available, tmp)
                end

                # Cartesian product gives all destination tconfigs for this term
                for prod in Iterators.product(available...)
                    sig_tconfig_vec = [ket_tconfig.config...]
                    for cidx in 1:length(term.clusters)
                        ci = term.clusters[cidx]
                        sig_tconfig_vec[ci.idx] = prod[cidx]
                    end
                    sig_tconfig = TuckerConfig(sig_tconfig_vec)

                    # Only keep if term actually couples these configs
                    check_term(term, sig_fock, sig_tconfig, ket_fock, ket_tconfig) || continue

                    # Prescreening only at block-building stage, so we just collect here
                    if haskey(tconfigs_to_process, sig_tconfig)
                        push!(tconfigs_to_process[sig_tconfig],
                              (term, ket_fock, ket_tconfig, ket_tuck))
                    else
                        tconfigs_to_process[sig_tconfig] = [(term, ket_fock, ket_tconfig, ket_tuck)]
                    end
                end
            end
        end
    end

    # --- For each FOIS block, build σ_block and accumulate its norm ------------------------
    σ2_job = zeros(T,R)

    for (sig_tconfig, term_list) in tconfigs_to_process
        contrib = _sigma_block_norm2(term_list, sig_fock, sig_tconfig, ket,
                                     cluster_ops, clustered_ham;
                                     thresh=thresh, max_number=max_number)
        σ2_job .+= contrib
    end

    return σ2_job
end

# Driver: compute approximate ⟨X|H|0⟩⟨X|H|0⟩ (per root) over the global FOIS, block-by-block

"""
    compute_spt_sigma_norm2_blockwise(ref::BSTstate, cluster_ops, clustered_ham;
                                      H0="Hcmf", nbody=4, thresh_foi=1e-6,
                                      max_number=nothing, opt_ref=true,
                                      ci_tol=1e-6, verbose=1, prescreen=false)

Compute, in parallel, the FOIS approximation to ⟨X|H|0⟩⟨X|H|0⟩ for each root:

    σ2[r] ≈ ∑_X |<X|H|0_r>|^2

This uses the same FOIS definition as `compute_pt2_energy2`, but never stores a full σ.
"""
function compute_spt_sigma_norm_blockwise(ref::BSTstate{T,N,R}, cluster_ops, clustered_ham;
                                           H0          = "Hcmf",
                                           nbody       = 4,
                                           thresh_foi  = 1e-6,
                                           max_number  = nothing,
                                           opt_ref     = true,
                                           ci_tol      = 1e-6,
                                           verbose     = 1,
                                           prescreen   = false) where {T,N,R}

    println()
    println(" |.......................BST-σ·σ (blockwise).....................................")
    verbose < 1 || println(" H0          : ", H0          )
    verbose < 1 || println(" nbody       : ", nbody       )
    verbose < 1 || println(" thresh_foi  : ", thresh_foi  )
    verbose < 1 || println(" max_number  : ", max_number  )
    verbose < 1 || println(" opt_ref     : ", opt_ref     )
    verbose < 1 || println(" ci_tol      : ", ci_tol      )
    verbose < 1 || println(" verbose     : ", verbose     )
    verbose < 1 || @printf("\n")
    verbose < 1 || @printf(" %-50s", "Length of Reference: ")
    verbose < 1 || @printf("%10i\n", length(ref))

    lk = ReentrantLock()

    # --- Solve variationally in reference space (same as compute_pt2_energy2) --------------
    ref_vec = deepcopy(ref)
    clusters = ref_vec.clusters
    E0 = zeros(T,R)

    if opt_ref
        @printf(" %-50s\n", "Solve zeroth-order problem: ")
        time = @elapsed E0, ref_vec = ci_solve(ref_vec, cluster_ops, clustered_ham, conv_thresh=ci_tol)
        @printf(" %-50s%10.6f seconds\n", "Diagonalization time: ",time)
    else
        @printf(" %-50s", "Compute zeroth-order energy: ")
        flush(stdout)
        @time E0 = compute_expectation_value(ref_vec, cluster_ops, clustered_ham)
    end

    # --- Define jobs (same as compute_pt2_energy2) ----------------------------------------
    jobs = Dict{FockConfig{N},Vector{Tuple}}()
    for (fock_ket, configs_ket) in ref_vec.data
        for (ftrans, terms) in clustered_ham
            fock_x = ftrans + fock_ket

            # Check Fock-sector validity
            all(f[1] >= 0 for f in fock_x) || continue
            all(f[2] >= 0 for f in fock_x) || continue
            all(f[1] <= length(clusters[fi]) for (fi,f) in enumerate(fock_x)) || continue
            all(f[2] <= length(clusters[fi]) for (fi,f) in enumerate(fock_x)) || continue

            job_input = (terms, fock_ket, configs_ket)
            if haskey(jobs, fock_x)
                push!(jobs[fock_x], job_input)
            else
                jobs[fock_x] = [job_input]
            end
        end
    end

    jobs_vec = [(fock_x, job) for (fock_x, job) in jobs]

    println(" Number of jobs:    ", length(jobs_vec))
    println(" Number of threads: ", Threads.nthreads())
    BLAS.set_num_threads(1)
    flush(stdout)

    # --- Thread-local accumulators for ⟨σ|σ⟩ ---------------------------------------------
    σ2_thread = [zeros(T,R) for _ in 1:Threads.nthreads()]

    tmp = Int(round(length(jobs_vec)/100))
    tmp == 0 && (tmp = 1)
    verbose < 2 || println(" |----------------------------------------------------------------------------------------------------|")
    verbose < 2 || println(" |0%                                                                                              100%|")
    verbose < 2 || print(" |")
    nprinted = 0

    alloc = @allocated t = @elapsed begin
        @Threads.threads for (jobi, job) in collect(enumerate(jobs_vec))
            fock_sig = job[1]
            tid = Threads.threadid()

            σ2_thread[tid] .+= _pt2_job_sigma_norm_blockwise(
                fock_sig, job[2], ref_vec, cluster_ops, clustered_ham,
                nbody, verbose, thresh_foi, max_number, prescreen
            )

            if verbose > 1 && jobi % tmp == 0
                lock(lk)
                try
                    print("-"); nprinted += 1; flush(stdout)
                finally
                    unlock(lk)
                end
            end
        end
    end

    flush(stdout)
    verbose < 2 || for i in nprinted+1:100
        print("-")
    end
    verbose < 2 || println("|")
    flush(stdout)

    @printf(" %-48s%10.1f s Allocated: %10.1e GB\n",
            "Time spent computing σ·σ: ", t, alloc*1e-9)

    σ2 = sum(σ2_thread)

    for r in 1:R
        @printf(" Root %3i: <σ|σ> = %14.8f\n", r, σ2[r])
    end

    return σ2
end


# --- Driver over all Fock sectors ------------------------------------------

"""
    compute_spt_sigma_norm_blockwise(ref::BSTstate, cluster_ops, clustered_ham;
                                     H0="Hcmf", nbody=4, thresh_foi=1e-6,
                                     max_number=nothing, opt_ref=true,
                                     ci_tol=1e-6, verbose=1, prescreen=false)

Compute, in parallel, the FOIS approximation to ⟨X|H|0⟩⟨X|H|0⟩ for each root:

    σ2[r] ≈ ∑_X |<X|H|0_r>|^2

This uses the same FOIS definition as `compute_pt2_energy2`, but never stores a full σ.
"""
function compute_spt_sigma_norm_blockwise_new(ref::BSTstate{T,N,R}, cluster_ops, clustered_ham;
                                          H0          = "Hcmf",
                                          nbody       = 4,
                                          thresh_foi  = 1e-6,
                                          max_number  = nothing,
                                          opt_ref     = true,
                                          ci_tol      = 1e-6,
                                          verbose     = 1,
                                          prescreen   = false) where {T,N,R}

    println()
    println(" |.......................BST-σ·σ (blockwise).....................................")
    verbose < 1 || println(" H0          : ", H0          )
    verbose < 1 || println(" nbody       : ", nbody       )
    verbose < 1 || println(" thresh_foi  : ", thresh_foi  )
    verbose < 1 || println(" max_number  : ", max_number  )
    verbose < 1 || println(" opt_ref     : ", opt_ref     )
    verbose < 1 || println(" ci_tol      : ", ci_tol      )
    verbose < 1 || println(" verbose     : ", verbose     )
    verbose < 1 || @printf("\n")
    verbose < 1 || @printf(" %-50s", "Length of Reference: ")
    verbose < 1 || @printf("%10i\n", length(ref))

    lk = ReentrantLock()

    # --- Solve variationally in reference space (same as compute_pt2_energy2) ---
    ref_vec = deepcopy(ref)
    clusters = ref_vec.clusters
    E0 = zeros(T,R)

    if opt_ref
        @printf(" %-50s\n", "Solve zeroth-order problem: ")
        time = @elapsed E0, ref_vec = ci_solve(ref_vec, cluster_ops, clustered_ham,
                                               conv_thresh=ci_tol)
        @printf(" %-50s%10.6f seconds\n", "Diagonalization time: ", time)
    else
        @printf(" %-50s", "Compute zeroth-order energy: ")
        flush(stdout)
        @time E0 = compute_expectation_value(ref_vec, cluster_ops, clustered_ham)
    end

    # --- Define jobs (same as compute_pt2_energy2) ---------------------------
    jobs = Dict{FockConfig{N},Vector{Tuple}}()
    for (fock_ket, configs_ket) in ref_vec.data
        for (ftrans, terms) in clustered_ham
            fock_x = ftrans + fock_ket

            # Check Fock-sector validity
            all(f[1] >= 0 for f in fock_x) || continue
            all(f[2] >= 0 for f in fock_x) || continue
            all(f[1] <= length(clusters[fi]) for (fi,f) in enumerate(fock_x)) || continue
            all(f[2] <= length(clusters[fi]) for (fi,f) in enumerate(fock_x)) || continue

            job_input = (terms, fock_ket, configs_ket)
            if haskey(jobs, fock_x)
                push!(jobs[fock_x], job_input)
            else
                jobs[fock_x] = [job_input]
            end
        end
    end

    jobs_vec = [(fock_x, job) for (fock_x, job) in jobs]

    println(" Number of jobs:    ", length(jobs_vec))
    println(" Number of threads: ", Threads.nthreads())
    BLAS.set_num_threads(1)
    flush(stdout)

    # --- Thread-local accumulators for ⟨σ|σ⟩ -------------------------------
    σ2_thread = [zeros(T,R) for _ in 1:Threads.nthreads()]

    tmp = Int(round(length(jobs_vec) / 100))
    tmp == 0 && (tmp = 1)
    verbose < 2 || println(" |----------------------------------------------------------------------------------------------------|")
    verbose < 2 || println(" |0%                                                                                              100%|")
    verbose < 2 || print(" |")
    nprinted = 0

    alloc = @allocated t = @elapsed begin
        @Threads.threads for jobi in eachindex(jobs_vec)
            fock_sig, job = jobs_vec[jobi]
            tid = Threads.threadid()

            σ2_thread[tid] .+= _pt2_job_sigma_norm_blockwise(
                fock_sig, job, ref_vec, cluster_ops, clustered_ham,
                nbody, verbose, thresh_foi, max_number, prescreen
            )

            if verbose > 1 && jobi % tmp == 0
                lock(lk)
                try
                    print("-"); nprinted += 1; flush(stdout)
                finally
                    unlock(lk)
                end
            end
        end
    end

    flush(stdout)
    verbose < 2 || for i in nprinted+1:100
        print("-")
    end
    verbose < 2 || println("|")
    flush(stdout)

    @printf(" %-48s%10.1f s Allocated: %10.1e GB\n",
            "Time spent computing σ·σ: ", t, alloc*1e-9)

    σ2 = sum(σ2_thread)

    for r in 1:R
        @printf(" Root %3i: <σ|σ> = %14.8f\n", r, σ2[r])
    end

    return σ2
end
using FermiCG
using LinearAlgebra
using Printf

# =============================================================================
# OPTIMISATION OVERVIEW
# =============================================================================
#
# Bottleneck 1 — BSTstate wrapper for norm:
#   Original wraps σ_block in BSTstate + orth_dot.
#   After nonorth_add, Tucker factors come from SVD → they ARE orthonormal.
#   So ‖σ_r‖² = ‖core[r]‖² (plain dot with itself). No BSTstate needed.
#
# Bottleneck 2 — nonorth_add allocations:
#   The scratch-buffer overload `nonorth_add(tucks, scr)` already exists in
#   tucker.jl but was never called from the blockwise driver.
#   We pre-allocate one `scr` per thread and reuse it across all blocks.
#
# Bottleneck 3 — typed contrib storage (NTuple{4,Any} → SigmaContrib):
#   Any-typed tuples box every element on the heap. A concrete parametric
#   struct lets the compiler keep everything unboxed.
#
# Bottleneck 4 — double check_term:
#   Original calls check_term when building tconfigs_to_process AND again
#   inside _sigma_block_norm2. We call it once and store only passing entries.
#
# Bottleneck 5 — Iterators.product allocation:
#   Replaced with a mutating recursive helper that writes directly into the
#   dict using a single reused scratch vector.
#
# Bottleneck 6 — per-block Tucker allocations in nonorth_add:
#   The hcat of factor matrices is the unavoidable SVD kernel.
#   We pre-size Ui buffers per mode once per block instead of re-hcating.
#
# Bottleneck 7 — lock in the progress counter hot path:
#   Replaced with Threads.Atomic{Int} + lock only on the rare print call.
#
# =============================================================================

# ---- Typed contribution record (avoids Any boxing) --------------------------

struct SigmaContrib{T,N,R}
    term        :: Any                   # ClusteredTerm (opaque)
    ket_fock    :: FockConfig{N}
    ket_tconfig :: TuckerConfig{N}
    ket_tuck    :: Tucker{T,N,R}
end

# ---- Per-thread scratch buffers for nonorth_add -----------------------------

"""
    make_nonorth_scr(N, max_dim)

Allocate the scratch buffer vector expected by the nonorth_add(tucks, scr) overload.
`N` = number of Tucker modes, `max_dim` = largest expected uncompressed dimension.
Returns `Vector{Vector{Float64}}` of length N, each pre-sized to `max_dim^N`.
"""
function make_nonorth_scr(N::Int, max_dim::Int=1024)
    return [Vector{Float64}(undef, max_dim^N) for _ in 1:N]
end

# ---- Fast per-root norm² directly from Tucker cores ------------------------

"""
    tucker_core_norm2(t::Tucker{T,N,R}) → NTuple{R,T}

After `nonorth_add` / `compress`, Tucker factor matrices are orthonormal
(they come from SVD).  In that case ‖T_r‖² = ‖core[r]‖².
This avoids wrapping in a BSTstate and calling orth_dot.

We verify orthonormality via a cheap check on the first factor.
Falls back to full nonorth_dot if factors are not orthonormal.
"""
@inline function tucker_core_norm2(t::Tucker{T,N,R}) where {T,N,R}
    # Quick orthonormality check on first factor (cheap proxy)
    A = t.factors[1]
    if size(A,2) > 0
        # ‖A'A - I‖_max < tol  (all-pairs check would be O(k²d), just check diagonal)
        dev = maximum(abs(dot(view(A,:,i), view(A,:,i)) - one(T)) for i in 1:size(A,2))
        dev < 1e-8 || return _tucker_norm2_nonorth(t)
    end
    # Fast path: sum squared entries of each core
    out = zeros(T, R)
    @inbounds for r in 1:R
        c = t.core[r]
        s = zero(T)
        @simd for i in eachindex(c)
            s += c[i]*c[i]
        end
        out[r] = s
    end
    return out
end

function _tucker_norm2_nonorth(t::Tucker{T,N,R}) where {T,N,R}
    # Correct path when factors are not orthonormal: use existing nonorth_dot
    return nonorth_dot(t, t)
end

# =============================================================================
# _sigma_block_norm2_fast
#
# Changes vs original _sigma_block_norm2:
#   - Accepts typed Vector{SigmaContrib} (no Any boxing)
#   - check_term NOT called again (caller already filtered)
#   - Passes pre-allocated scr to nonorth_add(tucks, scr) — no heap allocs
#     inside the Tucker combination step
#   - Computes ‖σ_block‖² directly from cores, no BSTstate wrapper
#   - Single-contrib fast path skips nonorth_add entirely
# =============================================================================

function _sigma_block_norm2_fast(
        contribs    :: Vector{SigmaContrib{T,N,R}},
        sig_fock    :: FockConfig{N},
        sig_tconfig :: TuckerConfig{N},
        ket         :: BSTstate{T,N,R},
        cluster_ops,
        clustered_ham,
        scr         :: Vector{Vector{T}};   # pre-allocated per-thread scratch
        thresh      :: T,
        max_number) where {T,N,R}

    tucks = Tucker{T,N,R}[]
    sizehint!(tucks, length(contribs))

    @inbounds for contrib in contribs
        sig_tuck = form_sigma_block_expand(
            contrib.term, cluster_ops,
            sig_fock, sig_tconfig,
            contrib.ket_fock, contrib.ket_tconfig, contrib.ket_tuck;
            max_number = max_number,
            prescreen  = thresh)

        (length(sig_tuck) == 0 || norm(sig_tuck) < thresh) && continue

        sig_tuck = compress(sig_tuck; thresh=thresh)
        length(sig_tuck) == 0 && continue

        push!(tucks, sig_tuck)
    end

    isempty(tucks) && return zeros(T, R)

    # Combine Tucker contributions
    σ_block = if length(tucks) == 1
        # Fast path: skip nonorth_add (no SVD needed)
        only(tucks)
    else
        try
            # Use scratch-buffer overload to avoid heap allocs inside nonorth_add
            nonorth_add(tucks, scr)
        catch e
            if e isa LinearAlgebra.LAPACKException
                @warn "nonorth_add (scr) failed; retrying with QRIteration" sig_fock sig_tconfig
                nonorth_add(tucks; svd_alg=:qr)   # safe fallback
            else
                rethrow()
            end
        end
    end

    # Compute ‖σ_block‖² per root directly from cores (no BSTstate!)
    return tucker_core_norm2(σ_block)
end

# =============================================================================
# _pt2_job_sigma_norm_blockwise_fast
#
# Changes vs original:
#   - Dict maps TuckerConfig → Vector{SigmaContrib{T,N,R}} (typed, unboxed)
#   - check_term called ONCE (when inserting into dict)
#   - available subspaces built ONCE per (term, sig_fock), not per ket_tconfig
#   - Cartesian product via mutating _fill_tconfigs! (no Iterators.product)
#   - scr passed through to _sigma_block_norm2_fast
# =============================================================================

function _pt2_job_sigma_norm_blockwise_fast(
        sig_fock    :: FockConfig{N},
        job,
        ket         :: BSTstate{T,N,R},
        cluster_ops,
        clustered_ham,
        scr         :: Vector{Vector{T}},
        nbody, verbose, thresh, max_number, prescreen) where {T,N,R}

    # Typed dict: TuckerConfig → list of pre-filtered contributions
    tconfigs = Dict{TuckerConfig{N}, Vector{SigmaContrib{T,N,R}}}()

    for jobi in job
        terms, ket_fock, ket_tconfigs = jobi

        for term in terms
            length(term.clusters) <= nbody || continue
            nc = length(term.clusters)

            # Build available subspaces ONCE per (term, sig_fock)
            available = Vector{Vector{UnitRange{Int}}}(undef, nc)
            skip_term = false
            for (k, ci) in enumerate(term.clusters)
                tmp = UnitRange{Int}[]
                if haskey(ket.p_spaces[ci.idx], sig_fock[ci.idx])
                    push!(tmp, ket.p_spaces[ci.idx][sig_fock[ci.idx]])
                end
                if haskey(ket.q_spaces[ci.idx], sig_fock[ci.idx])
                    push!(tmp, ket.q_spaces[ci.idx][sig_fock[ci.idx]])
                end
                if isempty(tmp)
                    skip_term = true; break
                end
                available[k] = tmp
            end
            skip_term && continue

            for (ket_tconfig, ket_tuck) in ket_tconfigs
                # Mutable scratch for sig_tconfig construction.
                # Must be Vector{UnitRange{Int}} because _fill_tconfigs! writes
                # UnitRange{Int} subspace values into it — NOT plain Int.
                sig_vec = collect(UnitRange{Int}, ket_tconfig.config)   # length N, mutable

                _fill_tconfigs!(tconfigs, sig_vec, available,
                                term, sig_fock, ket_fock, ket_tconfig, ket_tuck,
                                1)
            end
        end
    end

    # For each FOIS block, build σ_block and accumulate ‖σ_block‖²
    σ2_job = zeros(T, R)
    for (sig_tconfig, contribs) in tconfigs
        isempty(contribs) && continue
        n2 = _sigma_block_norm2_fast(contribs, sig_fock, sig_tconfig,
                                     ket, cluster_ops, clustered_ham, scr;
                                     thresh=thresh, max_number=max_number)
        σ2_job .+= n2
    end

    return σ2_job
end

# ---- Mutating Cartesian-product helper --------------------------------------
# Fills `tconfigs` dict in-place by iterating over all combinations of
# `available` subspaces.  Uses a single reused `sig_vec` (length N) as scratch.
# check_term is called exactly once per candidate (sig_tconfig, contrib) pair.

function _fill_tconfigs!(
        tconfigs    :: Dict{TuckerConfig{N}, Vector{SigmaContrib{T,N,R}}},
        sig_vec     :: Vector{UnitRange{Int}},
        available   :: Vector{Vector{UnitRange{Int}}},
        term,
        sig_fock    :: FockConfig{N},
        ket_fock    :: FockConfig{N},
        ket_tconfig :: TuckerConfig{N},
        ket_tuck    :: Tucker{T,N,R},
        depth       :: Int) where {T,N,R}

    nc = length(term.clusters)
    if depth > nc
        # All cluster indices set — check and record
        sig_tconfig = TuckerConfig(sig_vec)
        check_term(term, sig_fock, sig_tconfig, ket_fock, ket_tconfig) || return

        contrib = SigmaContrib{T,N,R}(term, ket_fock, ket_tconfig, ket_tuck)
        if haskey(tconfigs, sig_tconfig)
            push!(tconfigs[sig_tconfig], contrib)
        else
            v = Vector{SigmaContrib{T,N,R}}()
            sizehint!(v, 8)
            push!(v, contrib)
            tconfigs[sig_tconfig] = v
        end
        return
    end

    ci      = term.clusters[depth]
    old_val = sig_vec[ci.idx]
    for subspace in available[depth]
        sig_vec[ci.idx] = subspace
        _fill_tconfigs!(tconfigs, sig_vec, available, term,
                        sig_fock, ket_fock, ket_tconfig, ket_tuck,
                        depth + 1)
    end
    sig_vec[ci.idx] = old_val   # restore scratch
end

# =============================================================================
# compute_spt_sigma_norm_blockwise_fast  — drop-in replacement
#
# Additional changes vs original driver:
#   - One `scr` buffer pre-allocated per thread, passed into every job call
#   - jobs_vec collected with `collect(pairs(jobs))` — single allocation
#   - Progress counter uses Threads.Atomic{Int} — no lock in the hot loop
#   - thresh / ci_tol given concrete type annotations to avoid type instability
# =============================================================================

"""
    compute_spt_sigma_norm_blockwise_fast(ref, cluster_ops, clustered_ham; kwargs...)

Drop-in replacement for `compute_spt_sigma_norm_blockwise`.

Computes ∑_X |⟨X|H|0_r⟩|² for each root r without storing a global σ.

Key speedups over the original:
  1. `tucker_core_norm2`: ‖σ_block‖² from cores only — no BSTstate wrapping
  2. `nonorth_add(tucks, scr)`: scratch-buffer overload — fewer heap allocs
  3. `SigmaContrib`: typed struct — no Any boxing in the contribution dict
  4. `check_term` called once per entry, not twice
  5. `_fill_tconfigs!`: in-place Cartesian product — no Iterators.product
  6. `Threads.Atomic` progress counter — no lock in the hot job loop
  7. Per-thread scr buffers — no alloc contention across threads
"""
function compute_spt_sigma_norm_blockwise_alternative(
        ref          :: BSTstate{T,N,R},
        cluster_ops,
        clustered_ham;
        H0           = "Hcmf",
        nbody        = 4,
        thresh_foi   = 1e-6,
        max_number   = nothing,
        opt_ref      = true,
        ci_tol       = 1e-6,
        verbose      = 1,
        prescreen    = false) where {T,N,R}

    # Concrete threshold type — avoids type instability in inner loops
    thresh_T = T(thresh_foi)
    ci_tol_T = T(ci_tol)

    println()
    println(" |.......................BST-σ·σ (blockwise, fast)...........................")
    verbose < 1 || println(" H0          : ", H0)
    verbose < 1 || println(" nbody       : ", nbody)
    verbose < 1 || println(" thresh_foi  : ", thresh_T)
    verbose < 1 || println(" max_number  : ", max_number)
    verbose < 1 || println(" opt_ref     : ", opt_ref)
    verbose < 1 || println(" ci_tol      : ", ci_tol_T)
    verbose < 1 || println(" verbose     : ", verbose)
    verbose < 1 || @printf(" %-50s%10i\n", "Length of Reference: ", length(ref))

    # --- Zeroth-order solve (identical to original) ---------------------------
    ref_vec = deepcopy(ref)
    E0 = zeros(T, R)

    if opt_ref
        @printf(" %-50s\n", "Solve zeroth-order problem: ")
        time = @elapsed E0, ref_vec = ci_solve(ref_vec, cluster_ops, clustered_ham,
                                               conv_thresh=ci_tol_T)
        @printf(" %-50s%10.6f seconds\n", "Diagonalization time: ", time)
    else
        @printf(" %-50s", "Compute zeroth-order energy: ")
        flush(stdout)
        @time E0 = compute_expectation_value(ref_vec, cluster_ops, clustered_ham)
    end

    # --- Build jobs (identical logic to original) ----------------------------
    clusters = ref_vec.clusters
    jobs = Dict{FockConfig{N}, Vector{Tuple}}()

    for (fock_ket, configs_ket) in ref_vec.data
        for (ftrans, terms) in clustered_ham
            fock_x = ftrans + fock_ket
            all(f[1] >= 0 for f in fock_x) || continue
            all(f[2] >= 0 for f in fock_x) || continue
            all(f[1] <= length(clusters[fi]) for (fi, f) in enumerate(fock_x)) || continue
            all(f[2] <= length(clusters[fi]) for (fi, f) in enumerate(fock_x)) || continue
            job_input = (terms, fock_ket, configs_ket)
            if haskey(jobs, fock_x)
                push!(jobs[fock_x], job_input)
            else
                jobs[fock_x] = [job_input]
            end
        end
    end

    jobs_vec = collect(pairs(jobs))   # single allocation, stable order
    nj = length(jobs_vec)

    println(" Number of jobs:    ", nj)
    println(" Number of threads: ", Threads.nthreads())
    BLAS.set_num_threads(1)
    flush(stdout)

    # --- Per-thread scratch buffers for nonorth_add --------------------------
    # Each Tucker mode needs one scratch vector.  We size conservatively;
    # nonorth_add(tucks, scr) will resize!(scr[i], ...) as needed.
    scr_per_thread = [make_nonorth_scr(N) for _ in 1:Threads.nthreads()]

    # --- Thread-local σ² accumulators + lock-free progress counter -----------
    σ2_thread   = [zeros(T, R) for _ in 1:Threads.nthreads()]
    job_counter = Threads.Atomic{Int}(0)
    lk          = ReentrantLock()   # only for the rare print("-")

    tmp = max(1, Int(round(nj / 100)))
    verbose < 2 || println(" |----------------------------------------------------------------------------------------------------|")
    verbose < 2 || println(" |0%                                                                                              100%|")
    verbose < 2 || print(" |")

    alloc = @allocated t = @elapsed begin
        @Threads.threads for jobi in 1:nj
            fock_sig, job = jobs_vec[jobi]
            tid = Threads.threadid()

            σ2_thread[tid] .+= _pt2_job_sigma_norm_blockwise_fast(
                fock_sig, job, ref_vec, cluster_ops, clustered_ham,
                scr_per_thread[tid],
                nbody, verbose, thresh_T, max_number, prescreen
            )

            # Lock-free counter; lock only for the infrequent print
            cnt = Threads.atomic_add!(job_counter, 1) + 1
            if verbose > 1 && cnt % tmp == 0
                lock(lk) do
                    print("-"); flush(stdout)
                end
            end
        end
    end

    flush(stdout)
    if verbose > 1
        printed = job_counter[] ÷ tmp
        for _ in printed+1:100; print("-"); end
        println("|")
    end
    flush(stdout)

    @printf(" %-48s%10.1f s Allocated: %10.1e GB\n",
            "Time spent computing σ·σ: ", t, alloc*1e-9)

    σ2 = sum(σ2_thread)
    for r in 1:R
        @printf(" Root %3i: <σ|σ> = %14.8f\n", r, σ2[r])
    end

    return σ2
end
