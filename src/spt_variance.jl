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
    σ_block = nonorth_add(tucks)  # Tucker{T,N,R}

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
function compute_spt_sigma_norm2_blockwise(ref::BSTstate{T,N,R}, cluster_ops, clustered_ham;
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
