import math

class rd:
    bits = 5
    count = 32

class rd_nz:
    bits = 5
    count = 31

class rs:
    bits = 5
    count = 32

class rs_imm:
    bits = 5
    count = 32

class rsd:
    bits = 5
    count = 32

class rsd_nz:
    bits = 5
    count = 31

class rd_3:
    bits = 3
    count = 8

class rs_3:
    bits = 3
    count = 8

class rsd_3:
    bits = 3
    count = 8

class imm3:
    bits = 3
    count = 8

class imm4:
    bits = 4
    count = 16

class imm5:
    bits = 5
    count = 32

class imm6:
    bits = 6
    count = 64

class imm8:
    bits = 8
    count = 256

class imm9:
    bits = 8
    count = 512

class imm10:
    bits = 10
    count = 1024

class imm11:
    bits = 11
    count = 2048

class set0:
    bits = 4
    count = 16
    display = ".set0  "
    values = {
        "addi0",
        "addi1",
        "subi0",
        "subi1",
        "andi0",
        "andi1",
        "bici0",
        "bici1",
        "add",
        "addw",
        "sub",
        "subw",
        "and",
        "bic",
        "or",
        "xor",
    }

class full:
    bits = 5
    count = 32
    display = ".full  "
    values = {
        "addi0",
        "addi1",
        "subi0",
        "subi1",
        "andi0",
        "andi1",
        "bici0",
        "bici1",
        "slli0",
        "slli1",
        "srli0",
        "srli1",
        "srai0",
        "srai1",
        "rsbi0",
        "rsbi1",
        "add",
        "addw",
        "sub",
        "subw",
        "and",
        "bic",
        "or",
        "xor",
        "mul",
        "mulh",
        "div",
        "rem",
        "??",
        "??",
        "??",
        "??",
    }

class more:
    bits = 2
    count = 4
    display = ".more  "
    values = {
        "addiw",
        "subiw",
        "addi4spn",
        "subi4spn",
    }

class ldst:
    bits = 4
    count = 16
    display = ".ldst  "
    values = {
        "lb",
        "lh",
        "lw",
        "ld",
        "lq",
        "sb",
        "sh",
        "sw",
        "sd",
        "sq",

        "lbu",
        "lhu",
        "lwu/flw",
        "ldu/fld",
        "fsw",
        "fsd",
    }

class SHL:
    bits = 0
    count = 1
    display = "<<k"

class SHL1:
    bits = 0
    count = 1
    display = "*2"

class SHL2:
    bits = 0
    count = 1
    display = "*4"

class SHL3:
    bits = 0
    count = 1
    display = "*8"

class SHL4:
    bits = 0
    count = 1
    display = "*16"

class SP:
    bits = 0
    count = 1

class RA:
    bits = 0
    count = 1

class RDp1:
    bits = 0
    count = 1
    display = " =Rd+1"

class RSp1:
    bits = 0
    count = 1
    display = " =Rs+1"

class RD2RS:
    bits = 0
    count = 1
    display = " =Rd"

class RS2RS:
    bits = 0
    count = 1
    display = " =Rs"

class RD2RSD:
    bits = 0
    count = 1
    display = " =Rd"

class RS2RSD:
    bits = 0
    count = 1
    display = " =Rs"

class IMMp1:
    bits = 0
    count = 1


class BR:
    bits = 0
    count = 1

def measure(instructions):
    size = 0
    for (name, fields) in instructions.items():
        opcodes = name.split(',')
        def fieldname(f):
            if f == BR:
                return f"   ; {opcodes.pop(0)}"
            if hasattr(f, 'display'):
                return f.display
            return f" {f.__name__}"

        bits = sum(map(lambda x: x.bits, fields))
        count = math.prod(map(lambda x: x.count, fields))
        display = "".join(map(fieldname, [BR]+fields))
        print(f"{size:#10x}: {name:<16}  {bits} bits, {count:#10x} values {display}")
        size += count
    print(f"total size: ({size:#x}),  bits: {(size - 1).bit_length()}")
    print()


rvc = {
    # Q0
    "addi4spn":     [ rd_3, SP, imm8 ],
    "fld/lq":       [ rd_3, rs_3, imm5 ],
    "lw":           [ rd_3, rs_3, imm5 ],
    "flw/ld":       [ rd_3, rs_3, imm5 ],
    "fsd/sq":       [ rs_3, rs_3, imm5 ],
    "sw":           [ rs_3, rs_3, imm5 ],
    "fsw/sd":       [ rs_3, rs_3, imm5 ],

    # Q1
    "nop":          [],
    "addi":         [ rsd_nz, imm6 ],
    "jal":          [ RA, imm11 ],
    "addiw":        [ rsd_nz, imm6 ],
    "li":           [ rsd_nz, imm6 ],
    "addi16sp":     [ SP, imm6 ],
    "lui":          [ rsd_nz, imm6 ],
    "srli/64":      [ rsd_3, imm6 ],
    "srai/64":      [ rsd_3, imm6 ],
    "andi":         [ rsd_3, imm6 ],
    "sub":          [ rsd_3, rs_3 ],
    "xor":          [ rsd_3, rs_3 ],
    "or":           [ rsd_3, rs_3 ],
    "and":          [ rsd_3, rs_3 ],
    "subw":         [ rsd_3, rs_3 ],
    "addw":         [ rsd_3, rs_3 ],
    "j":            [ imm11 ],
    "beqz":         [ rs_3, imm8 ],
    "bnez":         [ rs_3, imm8 ],

    # Q2
    "slli/64":      [ rsd_nz, imm6 ],
    "fldsp/lqsp":   [ rd_nz, SP, imm6 ],
    "lwsp":         [ rd_nz, SP, imm6 ],
    "flwsp/ldsp":   [ rd_nz, SP, imm6 ],
    "jr":           [ rs ],
    "mv":           [ rd_nz, rs ],
    "ebreak":       [ ],
    "jalr":         [ RA, rs ],
    "add":          [ rsd_nz, rs ],
    "fsdsp/sqsp":   [ rs, SP, imm6 ],
    "swsp":         [ rs, SP, imm6 ],
    "fswsp/sdsp":   [ rs, SP, imm6 ],
}

my_attempt = {
# no sharing
    "ari,ari":          [ set0, rsd, rs_imm,        BR, set0, rsd, rs_imm ],

# share Rs1:
    "ari,ari,(rs)":     [ full, rsd, rs_imm,        BR, full, RS2RSD, rs_imm ],

# forward Rd to Rs1
    "ari,ari,(rd)":     [ full, rsd, rs_imm,        BR, full, RD2RSD, rs_imm ],

# share Rs1 and Rs2/Imm
    #"ari,beqz":         [ set0, rsd, rs_imm,        BR, rs, imm6 ],
    #"ari,bnez":         [ set0, rsd, rs_imm,        BR, rs, imm6 ],

    "ari,beqz,(rd)":    [ set0, rsd, rs_imm,        BR, RD2RS, imm11 ],
    "ari,bnez,(rd)":    [ set0, rsd, rs_imm,        BR, RD2RS, imm11 ],

    "ari,j":            [ full, rsd, rs_imm,        BR, imm10 ],
    "ari,jal":          [ full, rsd, rs_imm,        BR, RA, imm10 ],

    "ar2,jr":           [ more, rd, rs, imm5,       BR, rs ],
    "ar2,jalr":         [ more, rd, rs, imm5,       BR, RA, rs ],
    "ari,jr":           [ full, rsd, rs_imm,        BR, rs ],
    "ari,jalr":         [ full, rsd, rs_imm,        BR, RA, rs ],

    "--resv21":         [ imm10, imm11 ],
    "mul,mulh":         [ rd, rs, rs,               BR, rd, RS2RS, RS2RS ],
    "div,rem":          [ rd, rs, rs,               BR, rd, RS2RS, RS2RS ],
    "add,sub":          [ rd, rs, rs,               BR, rd, RS2RS, RS2RS ],
    "and,bic":          [ rd, rs, rs,               BR, rd, RS2RS, RS2RS ],
    "--resv25":         [ imm10, imm10, imm5 ],

    "mem,mem,(rsimm)":  [ ldst, rd, rs, imm10,SHL,  BR, RDp1, RS2RS, IMMp1,SHL ],

# independent but loads have no offset (presumed calculated in adjacent op, but not mandatory)
    "ari,mem":          [ set0, rsd, rs_imm,SHL,    BR, ldst, rd, rs ],
    "mem,ari":          [ ldst, rd, rs,             BR, set0, rsd, rs_imm,SHL ],
}

measure(rvc)
measure(my_attempt)
