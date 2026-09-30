"""
    space(::SiteType"Hole";
          conserve_qns = false,
          conserve_sz = conserve_qns,
          conserve_nf = conserve_qns,
          conserve_nfparity = conserve_qns,
          qnname_sz = "Sz",
          qnname_nf = "Nh",
          qnname_nfparity = "NhParity")

Create the Hilbert space for a site of type "Hole".

Optionally specify the conserved symmetries and their quantum number labels.
"""

using ITensors
import ITensors: space, val, state, op, has_fermion_string, alias
import ITensors: SiteType, OpName, StateName, ValName
function space(
    ::SiteType"Hole";
    conserve_qns=false,
    conserve_sz=conserve_qns,
    conserve_nf=conserve_qns,
    conserve_nfparity=conserve_qns,
    qnname_sz="Sz",
    qnname_nf="Nh",
    qnname_nfparity="NhParity",
    # Deprecated
    conserve_parity=nothing
)
    if !isnothing(conserve_parity)
        conserve_nfparity = conserve_parity
    end
    if conserve_sz && conserve_nf
        return [
            QN((qnname_nf, 0, -1), (qnname_sz, 0)) => 1
            QN((qnname_nf, 1, -1), (qnname_sz, +1)) => 1
            QN((qnname_nf, 1, -1), (qnname_sz, -1)) => 1
            QN((qnname_nf, 2, -1), (qnname_sz, 0)) => 1
        ]
    elseif conserve_nf
        return [
            QN(qnname_nf, 0, -1) => 1
            QN(qnname_nf, 1, -1) => 2
            QN(qnname_nf, 2, -1) => 1
        ]
    elseif conserve_sz
        return [
            QN((qnname_sz, 0), (qnname_nfparity, 0, -2)) => 1
            QN((qnname_sz, +1), (qnname_nfparity, 1, -2)) => 1
            QN((qnname_sz, -1), (qnname_nfparity, 1, -2)) => 1
            QN((qnname_sz, 0), (qnname_nfparity, 0, -2)) => 1
        ]
    elseif conserve_nfparity
        return [
            QN(qnname_nfparity, 0, -2) => 1
            QN(qnname_nfparity, 1, -2) => 2
            QN(qnname_nfparity, 0, -2) => 1
        ]
    end
    return 4
end

val(::ValName"Emp", ::SiteType"Hole") = 1
val(::ValName"Up", ::SiteType"Hole") = 2
val(::ValName"Dn", ::SiteType"Hole") = 3
val(::ValName"UpDn", ::SiteType"Hole") = 4
val(::ValName"0", st::SiteType"Hole") = val(ValName("Emp"), st)
val(::ValName"↑", st::SiteType"Hole") = val(ValName("Up"), st)
val(::ValName"↓", st::SiteType"Hole") = val(ValName("Dn"), st)
val(::ValName"↑↓", st::SiteType"Hole") = val(ValName("UpDn"), st)

state(::StateName"Emp", ::SiteType"Hole") = [1.0, 0, 0, 0]
state(::StateName"Up", ::SiteType"Hole") = [0.0, 1, 0, 0]
state(::StateName"Dn", ::SiteType"Hole") = [0.0, 0, 1, 0]
state(::StateName"UpDn", ::SiteType"Hole") = [0.0, 0, 0, 1]
state(::StateName"0", st::SiteType"Hole") = state(StateName("Emp"), st)
state(::StateName"↑", st::SiteType"Hole") = state(StateName("Up"), st)
state(::StateName"↓", st::SiteType"Hole") = state(StateName("Dn"), st)
state(::StateName"↑↓", st::SiteType"Hole") = state(StateName("UpDn"), st)

function op(::OpName"Nup", ::SiteType"Hole")
    return [
        0.0 0.0 0.0 0.0
        0.0 1.0 0.0 0.0
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 1.0
    ]
end
function op(on::OpName"n↑", st::SiteType"Hole")
    return op(alias(on), st)
end

function op(::OpName"Ndn", ::SiteType"Hole")
    return [
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
        0.0 0.0 1.0 0.0
        0.0 0.0 0.0 1.0
    ]
end
function op(on::OpName"n↓", st::SiteType"Hole")
    return op(alias(on), st)
end

function op(::OpName"Nupdn", ::SiteType"Hole")
    return [
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 1.0
    ]
end
function op(on::OpName"n↑↓", st::SiteType"Hole")
    return op(alias(on), st)
end

function op(::OpName"Ntot", ::SiteType"Hole")
    return [
        0.0 0.0 0.0 0.0
        0.0 1.0 0.0 0.0
        0.0 0.0 1.0 0.0
        0.0 0.0 0.0 2.0
    ]
end
function op(on::OpName"ntot", st::SiteType"Hole")
    return op(alias(on), st)
end

function op(::OpName"Cup", ::SiteType"Hole")
    return [
        0.0 1.0 0.0 0.0
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 1.0
        0.0 0.0 0.0 0.0
    ]
end
function op(on::OpName"c↑", st::SiteType"Hole")
    return op(alias(on), st)
end

function op(::OpName"Cdagup", ::SiteType"Hole")
    return [
        0.0 0.0 0.0 0.0
        1.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
        0.0 0.0 1.0 0.0
    ]
end
function op(on::OpName"c†↑", st::SiteType"Hole")
    return op(alias(on), st)
end

function op(::OpName"Cdn", ::SiteType"Hole")
    return [
        0.0 0.0 1.0 0.0
        0.0 0.0 0.0 -1.0
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
    ]
end
function op(on::OpName"c↓", st::SiteType"Hole")
    return op(alias(on), st)
end

function op(::OpName"Cdagdn", ::SiteType"Hole")
    return [
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
        1.0 0.0 0.0 0.0
        0.0 -1.0 0.0 0.0
    ]
end
function op(::OpName"c†↓", st::SiteType"Hole")
    return op(OpName("Cdagdn"), st)
end

function op(::OpName"Aup", ::SiteType"Hole")
    return [
        0.0 1.0 0.0 0.0
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 1.0
        0.0 0.0 0.0 0.0
    ]
end
function op(::OpName"a↑", st::SiteType"Hole")
    return op(OpName("Aup"), st)
end

function op(::OpName"Adagup", ::SiteType"Hole")
    return [
        0.0 0.0 0.0 0.0
        1.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
        0.0 0.0 1.0 0.0
    ]
end
function op(::OpName"a†↑", st::SiteType"Hole")
    return op(OpName("Adagup"), st)
end

function op(::OpName"Adn", ::SiteType"Hole")
    return [
        0.0 0.0 1.0 0.0
        0.0 0.0 0.0 1.0
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
    ]
end
function op(::OpName"a↓", st::SiteType"Hole")
    return op(OpName("Adn"), st)
end

function op(::OpName"Adagdn", ::SiteType"Hole")
    return [
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
        1.0 0.0 0.0 0.0
        0.0 1.0 0.0 0.0
    ]
end
function op(::OpName"a†↓", st::SiteType"Hole")
    return op(OpName("Adagdn"), st)
end

function op(::OpName"F", ::SiteType"Hole")
    return [
        1.0 0.0 0.0 0.0
        0.0 -1.0 0.0 0.0
        0.0 0.0 -1.0 0.0
        0.0 0.0 0.0 1.0
    ]
end

function op(::OpName"Fup", ::SiteType"Hole")
    return [
        1.0 0.0 0.0 0.0
        0.0 -1.0 0.0 0.0
        0.0 0.0 1.0 0.0
        0.0 0.0 0.0 -1.0
    ]
end
function op(::OpName"F↑", st::SiteType"Hole")
    return op(OpName("Fup"), st)
end

function op(::OpName"Fdn", ::SiteType"Hole")
    return [
        1.0 0.0 0.0 0.0
        0.0 1.0 0.0 0.0
        0.0 0.0 -1.0 0.0
        0.0 0.0 0.0 -1.0
    ]
end
function op(::OpName"F↓", st::SiteType"Hole")
    return op(OpName("Fdn"), st)
end

function op(::OpName"Sz", ::SiteType"Hole")
    #Op[s' => 2, s => 2] = +0.5
    #return Op[s' => 3, s => 3] = -0.5
    return [
        0.0 0.0 0.0 0.0
        0.0 0.5 0.0 0.0
        0.0 0.0 -0.5 0.0
        0.0 0.0 0.0 0.0
    ]
end

function op(::OpName"Sᶻ", st::SiteType"Hole")
    return op(OpName("Sz"), st)
end

function op(::OpName"Sx", ::SiteType"Hole")
    return [
        0.0 0.0 0.0 0.0
        0.0 0.0 0.5 0.0
        0.0 0.5 0.0 0.0
        0.0 0.0 0.0 0.0
    ]
end

function op(::OpName"Sˣ", st::SiteType"Hole")
    return op(OpName("Sx"), st)
end

function op(::OpName"S+", ::SiteType"Hole")
    return [
        0.0 0.0 0.0 0.0
        0.0 0.0 1.0 0.0
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
    ]
end

function op(::OpName"S⁺", st::SiteType"Hole")
    return op(OpName("S+"), st)
end
function op(::OpName"Sp", st::SiteType"Hole")
    return op(OpName("S+"), st)
end
function op(::OpName"Splus", st::SiteType"Hole")
    return op(OpName("S+"), st)
end

function op(::OpName"S-", ::SiteType"Hole")
    return [
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
        0.0 1.0 0.0 0.0
        0.0 0.0 0.0 0.0
    ]
end

function op(::OpName"S⁻", st::SiteType"Hole")
    return op(OpName("S-"), st)
end
function op(::OpName"Sm", st::SiteType"Hole")
    return op(OpName("S-"), st)
end
function op(::OpName"Sminus", st::SiteType"Hole")
    return op(OpName("S-"), st)
end

has_fermion_string(::OpName"Cup", ::SiteType"Hole") = true
function has_fermion_string(on::OpName"c↑", st::SiteType"Hole")
    return has_fermion_string(alias(on), st)
end
has_fermion_string(::OpName"Cdagup", ::SiteType"Hole") = true
function has_fermion_string(on::OpName"c†↑", st::SiteType"Hole")
    return has_fermion_string(alias(on), st)
end
has_fermion_string(::OpName"Cdn", ::SiteType"Hole") = true
function has_fermion_string(on::OpName"c↓", st::SiteType"Hole")
    return has_fermion_string(alias(on), st)
end
has_fermion_string(::OpName"Cdagdn", ::SiteType"Hole") = true
function has_fermion_string(on::OpName"c†↓", st::SiteType"Hole")
    return has_fermion_string(alias(on), st)
end