using LinearAlgebra

# 定义 Diquark_condensate_measurement 结构体
mutable struct Diquark_condensate_measurement{Dim,TG,TD,TF,TF_vec,TCov} <: AbstractMeasurement
    filename::Union{Nothing,String}
    _temporary_gaugefields::Temporalfields{TG}
    Dim::Int8
    verbose_print::Union{Verbose_print,Nothing}
    printvalues::Bool
    D::TD
    fermi_action::TF
    _temporary_fermionfields::Vector{TF_vec}
    Nr::Int64
    factor::Float64
    order::Int64
    cov_neural_net::TCov

    function Diquark_condensate_measurement(
        U::Vector{T};
        filename=nothing,
        verbose_level=2,
        printvalues=false,
        fermiontype="Staggered",
        mass=0.1,
        Nf=2,
        κ=1,
        r=1,
        L5=2,
        M=-1,
        eps_CG=1e-14,
        MaxCGstep=5000,
        BoundaryCondition=nothing,
        Nr=10,
        order=1,
        cov_neural_net=nothing
    ) where {T}

        Dim = length(U)
        if BoundaryCondition == nothing
            if Dim == 4
                boundarycondition = BoundaryCondition_4D_default
            elseif Dim == 2
                boundarycondition = BoundaryCondition_2D_default
            end
        else
            boundarycondition = BoundaryCondition
        end

        params, parameters_action, x, factor = make_fermionparameter_dict(U,
            fermiontype, mass,
            Nf,
            κ,
            r,
            L5,
            M,
        )

        params["eps_CG"] = eps_CG
        params["verbose_level"] = verbose_level
        params["MaxCGstep"] = MaxCGstep
        params["boundarycondition"] = boundarycondition

        D = Dirac_operator(U, x, params)
        fermi_action = FermiAction(D, parameters_action)
        TD = typeof(D)
        TF = typeof(fermi_action)

        TCov = typeof(cov_neural_net)

        myrank = get_myrank(U)
        if printvalues
            verbose_print = Verbose_print(verbose_level, myid=myrank, filename=filename)
        else
            verbose_print = nothing
        end

        numg = 1
        _temporary_gaugefields = Temporalfields(U[1], num=numg)

        numf = 2
        if order > 1
            numf += 1
        end

        TF_vec = typeof(x)
        _temporary_fermionfields = Vector{TF_vec}(undef, numf)
        for i = 1:numf
            _temporary_fermionfields[i] = similar(x)
        end

        return new{Dim,T,TD,TF,TF_vec,TCov}(
            filename,
            _temporary_gaugefields,
            Dim,
            verbose_print,
            printvalues,
            D,
            fermi_action,
            _temporary_fermionfields,
            Nr,
            factor,
            order,
            cov_neural_net
        )

    end
end

# 另一个 Diquark_condensate_measurement 构造函数
function Diquark_condensate_measurement(
    U::Vector{T},
    params::DiquarkCondensate_parameters,
    filename="Diquark_condensate.txt",
) where {T}

    if params.smearing_for_fermion == "nothing"
        cov_neural_net = nothing
    elseif params.smearing_for_fermion == "stout"
        cov_neural_net = CovNeuralnet(U)
        if params.stout_numlayers == 1
            st = STOUT_Layer(params.stout_loops, params.stout_ρ, U)
            push!(cov_neural_net, st)
        else
            if length(params.stout_ρ) == 1
                @warn "num. of stout layer is $(params.stout_numlayers) but there is only one rho. rho values are all same."
                for ilayer = 1:length(params.stout_ρ)
                    st = STOUT_Layer(params.stout_loops, params.stout_ρ, U)
                    push!(cov_neural_net, st)
                end
            else
                for ilayer = 1:length(params.stout_ρ)
                    st = STOUT_Layer(params.stout_loops, params.stout_ρ[ilayer], U)
                    push!(cov_neural_net, st)
                end
            end
        end
    else
        error("params.smearing_for_fermion = $(params.smearing_for_fermion) is not supported")
    end

    if params.fermiontype == "Staggered"
        method = Diquark_condensate_measurement(
            U;
            filename=filename,
            verbose_level=params.verbose_level,
            printvalues=params.printvalues,
            fermiontype=params.fermiontype,
            mass=params.mass,
            Nf=params.Nf,
            eps_CG=params.eps,
            MaxCGstep=params.MaxCGstep,
            Nr=params.Nr,
            cov_neural_net=cov_neural_net,
        )
    else
        error("$(params.fermiontype) is not supported in Diquark_condensate_measurement")
    end

    return method
end

# 测量二夸克凝聚的函数
function measure(
    m::M,
    U::Array{<:AbstractGaugefields{NC,Dim},1};
    additional_string="",
) where {M<:Diquark_condensate_measurement,NC,Dim}
    temps_fermi = get_temporary_fermionfields(m)
    p = temps_fermi[1]
    r = temps_fermi[2]

    if m.cov_neural_net === nothing
        D = m.D(U)
    else
        Uout, Uout_multi, _ = calc_smearedU(U, m.cov_neural_net)
        println("smeared U is used in diquark measurement")
        D = m.D(Uout)
    end

    ddc = 0.0
    Nr = m.Nr
    measurestring = ""
    if m.order != 1
        tmps = zeros(ComplexF64, m.order)
        p2 = temps_fermi[3]
        ddcs = zeros(ComplexF64, m.order)
    end

    for ir = 1:Nr
        clear_fermion!(p)
        Z4_distribution_fermi!(r)
        solve_DinvX!(p, D, r)
        # 这里需要根据二夸克凝聚的具体定义修改计算方式
        tmp = calculate_diquark(p) 
        if m.order != 1
            tmps[1] = tmp
            for i = 2:m.order
                solve_DinvX!(p2, D, p)
                p, p2 = p2, p
                tmps[i] = calculate_diquark(p)
            end
            ddcs .+= tmps
        end

        if m.printvalues
            measurestring_ir = "# $ir $additional_string $(real(tmp)/U[1].NV) # itrj irand diquarkcond"
            if m.order != 1
                measurestring_ir = "# $ir $additional_string"
                for i = 1:m.order
                    measurestring_ir *= " $(real(tmps[i])/U[1].NV) "
                end
                measurestring_ir *= " # itrj irand diquarkcond: $(m.order)-th orders"
            end
            println_verbose_level2(m.verbose_print, measurestring_ir)
            measurestring *= measurestring_ir * "\n"
        end
        ddc += tmp
    end

    ddc_value = real(ddc / Nr) / U[1].NV * m.factor
    if m.order != 1
        ddc_values = real.(ddcs / Nr) / U[1].NV * m.factor
    end

    if m.printvalues
        measurestring_ir = "$ddc_value # ddc Nr=$Nr"
        if m.order != 1
            measurestring_ir = " "
            for i = 1:m.order
                measurestring_ir *= " $(ddc_values[i]) "
            end
            measurestring_ir *= "# ddc Nr=$Nr"
        end
        println_verbose_level1(m.verbose_print, measurestring_ir)
        measurestring *= measurestring_ir * "\n"
        flush(stdout)
    end

    if m.order != 1
        output = Measurement_output(ddc_values, measurestring)
    else
        output = Measurement_output(ddc_value, measurestring)
    end

    return output
end

# 计算二夸克凝聚的函数，需要根据具体物理定义实现
function calculate_diquark(p)
    # 这里需要根据二夸克凝聚的具体定义实现计算逻辑
    # 例如，可能涉及到费米子场的张量积等操作
    return 0.0 
end    