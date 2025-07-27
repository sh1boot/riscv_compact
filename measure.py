import math

formats = {
    "nop": "nop",
    "---": "--reserved--",
    "pair.a":   "pair.a  {rd}, {rs1}, {rs2}",
    "pair.b":   "pair.b  {rd}, {rs1}, {rs2}",
    "add":      "add     {rd}, {rs1}, {rs2}",
    "addi0":    "addi    {rd}, {rs1}, #{imm}",
    "addi04spn":"addi    {rd}, SP, #{imm}*4",
    "addi1":    "addi    {rd}, {rs1}, #{imm}-32",
    "addi14spn":"addi    {rd}, SP, #{imm}*4+128",
    "addi14spn":"addi    {rd}, SP, #{imm}*4+256",
    "addi0w":   "addiw   {rd}, {rs1}, #{imm}",
    "addi1w":   "addiw   {rd}, {rs1}, #{imm}-32",
    "addw":     "addw    {rd}, {rs1}, {rs2}",
    "and":      "and     {rd}, {rs1}, {rs2}",
    "andi0":    "andi    {rd}, {rs1}, #{imm}",
    "andi1":    "andi    {rd}, {rs1}, #{imm}+32",
    "beqz":     "beqz    {rs1}, +{imm}",
    "bnez":     "bnez    {rs1}, +{imm}",
    "bic":      "bic     {rd}, {rs1}, {rs2}",
    "bittesti0":"andi    {rd}, {rs1}, #1<<{rs_imm}",
    "bittesti1":"andi    {rd}, {rs1}, #1<<{rs_imm}+32",
    "div":      "div     {rd}, {rs1}, {rs2}",
    "ebreak":   "ebreak",
    "fsd":      "fsd     {rd}, {imm}*16({rs1})",
    "fsw":      "fsw     {rd}, {imm}*4({rs1})",
    "j":        "j       +{imm}",
    "jr":       "jr      {rs2}",
    "jal":      "jal     {rd}, +{imm}",
    "jalr":     "jalr    {rd}, {rs2}",
    "lb":       "lb      {rd}, {imm}({rs1})",
    "lbu":      "lbu     {rd}, {imm}({rs1})",
    "ld":       "ld      {rd}, {imm}*8({rs1})",
    "ldu/fld":  "ldu     {rd}, {imm}*8({rs1})",
    "lh":       "lh      {rd}, {imm}*2({rs1})",
    "lhu":      "lhu     {rd}, {imm}*2({rs1})",
    "lq":       "lq      {rd}, {imm}*4({rs1})",
    "lw":       "lw      {rd}, {imm}*4({rs1})",
    "lwu/flw":  "lwu     {rd}, {imm}*4({rs1})",
    "mul":      "Mul     {rd}, {rs1}, {rs2}",
    "mulh":     "mulh    {rd}, {rs1}, {rs2}",
    "mv":       "mv      {rd}, {rs2}",
    "or":       "or      {rd}, {rs1}, {rs2}",
    "rem":      "rem     {rd}, {rs1}, {rs2}",
    "rsbi0":    "subi    {rd}, #{imm}, {rs1}",
    "rsbi1":    "subi    {rd}, #{imm}-32, {rs1}",
    "sb":       "sb      {rd}, {imm}({rs1})",
    "sd":       "sd      {rd}, {imm}*8({rs1})",
    "seqi0":    "slti    {rd}, {rs1}, #{rs_imm}",
    "seqi1":    "slti    {rd}, {rs1}, #{rs_imm}",
    "sh":       "sh      {rd}, {imm}*2({rs1})",
    "sll":      "sll     {rd}, {rs1}, {rs2}",
    "slli0":    "slli    {rd}, {rs1}, #{imm}",
    "slli1":    "slli    {rd}, {rs1}, #{imm}-32",
    "slti0":    "slti    {rd}, {rs1}, #{rs_imm}",
    "slti0u":   "slti    {rd}, {rs1}, #{rs_imm}",
    "slti1":    "slti    {rd}, {rs1}, #{rs_imm}+32",
    "slti1u":   "slti    {rd}, {rs1}, #{rs_imm}+32",
    "sq":       "sq      {rd}, {imm}*16({rs1})",
    "sra":      "sra     {rd}, {rs1}, {rs2}",
    "srai0":    "srai    {rd}, {rs1}, #{imm}",
    "srai1":    "srai    {rd}, {rs1}, #{imm}-32",
    "srl":      "srl     {rd}, {rs1}, {rs2}",
    "srli0":    "srli    {rd}, {rs1}, #{imm}",
    "srli1":    "srli    {rd}, {rs1}, #{imm}-32",
    "sub":      "sub     {rd}, {rs1}, {rs2}",
    "subw":     "subw    {rd}, {rs1}, {rs2}",
    "sw":       "sw      {rd}, {imm}*4({rs1})",
    "xor":      "xor     {rd}, {rs1}, {rs2}",
}

class ari3:
    bits = 3
    count = 8
    fmt = "arith3  {rd}, {rs1}, {rs_imm}"
    opcodes = {
        "addi0",
        "addi1",
        "subi0",
        "subi1",
        "add",
        "sub",
        "and",
        "or",
    }

class ari4:
    bits = 4
    count = 16
    fmt = "arith4  {rd}, {rs1}, {rs_imm}"
    opcodes = {
        "addi0",
        "addi1",
        "addi0w",
        "addi1w",
        "addi04spn",
        "addi14spn",
        "andi0",
        "andi1",
        "add",
        "addw",
        "sub",
        "subw",
        "and",
        "bic",
        "or",
        "xor",
    }

class ari5i:
    bits = 4
    count = 16
    fmt = "arith5i {rd}, {rs1}, {rs_imm}"
    opcodes = {
        "addi0",
        "addi1",
        "addi0w",
        "addi1w",
        "addi04spn",
        "addi14spn",
        "andi0",
        "andi1",
        "slli0",
        "slli1",
        "srli0",
        "srli1",
        "srai0",
        "srai1",
        "rsbi0",
        "rsbi1",
    }

class ari5r:
    bits = 4
    count = 16
    fmt = "arith5r {rd}, {rs1}, {rs2}"
    opcodes = {
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
        "sll",
        "srl",
        "sra",
        "???",
    }

class ari5x:
    bits = 5
    count = ari5r.count
    fmt = "arith5x {rd}, {rs1}, {rs_imm}"

class ari5:
    bits = 5
    count = ari5i.count + ari5r.count
    fmt = "arith5  {rd}, {rs1}, {rs_imm}"
    opcodes = ari5i.opcodes | ari5r.opcodes

class cmp:
    bits = 3
    count = 8
    fmt = "cmp     {rd}, {rs1}, {rs_imm}"
    opcodes = {
        "slti0",
        "slti1",
        "slti0u",
        "slti1u",
        "seqi0",
        "seqi1",
        "bittesti0",
        "bittesti1",
    }

class ldst:
    bits = 4
    count = 16
    fmt = "ldst    {rd}, {imm}({rs1})"
    opcodes = {
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

class IMPLICIT:
    __name__ = "-"
    bits = 0
    count = 1

class LDST(IMPLICIT):
    fmt = "=LdSt   {rd}, {imm}*k({rs1})"

class SHL(IMPLICIT):
    suffix = "<<k"

class pair:
    bits = 4
    count = 12
    fmt = "pair.a  {rd}, {rs1}, {rs2}"
    opcode_pairs = {
        ("add" ,"sltu*"),
        ("and", "bic"),
        ("min", "max"),
        ("minu", "maxu"),
        ("add", "sub"),
        ("mul", "mulhsu"),
        ("mul", "mulh"),
        ("mul", "mulhu"),
        ("div", "rem"),
        ("divu", "remu"),
        ("???", "???"),
        ("???", "???"),
    }
    opcodes = [p[0] for p in opcode_pairs]

class PAIR(IMPLICIT):
    fmt = "pair.b  {rd}, {rs1}, {rs2}"
    opcodes = [p[1] for p in pair.opcode_pairs]

class REGISTER:
    bits = 5
    count = 32

class REGISTER3:
    bits = 3
    count = 8

class rd(REGISTER):
    rd = "rd"

class rd_nz(REGISTER):
    count = 31
    rd = "rd"

class rsd(REGISTER):
    rd = "rsd"
    rs1 = "rsd"

class rsd_nz(REGISTER):
    count = 31
    rd = "rsd"
    rs1 = "rsd"

class rs1(REGISTER):
    rs1 = "rs1"

class rs2(REGISTER):
    rs2 = "rs2"
    rs2_imm = rs2

class rs_imm:
    bits = 5
    count = 32
    rs2 = "rs2"
    imm = "imm"
    rs_imm = "rs_imm"

class rd_3(REGISTER3):
    rd = "rd"

class rsd_3(REGISTER3):
    rd = "rsd"
    rs1 = "rsd"

class rs1_3(REGISTER3):
    rs1 = "rs1"

class rs2_3(REGISTER3):
    rs2 = "rs2"

class imm0(IMPLICIT):
    imm = "0"

class imm3:
    bits = 3
    count = 8
    imm = "imm3"

class imm4:
    bits = 4
    count = 16
    imm = "imm4"

class imm5:
    bits = 5
    count = 32
    imm = "imm5"

class imm6:
    bits = 6
    count = 64
    imm = "imm6"

class imm8:
    bits = 8
    count = 256
    imm = "imm8"

class imm9:
    bits = 9
    count = 512
    imm = "imm9"

class imm10:
    bits = 10
    count = 1024
    imm = "imm10"

class imm11:
    bits = 11
    count = 2048
    imm = "imm11"

class RD(IMPLICIT):
    def __init__(self, src): self.rd = f"={src}"

class RSD(IMPLICIT):
    def __init__(self, src):
        self.rd = f"={src}"
        self.rs1 = self.rd

class RS1(IMPLICIT):
    def __init__(self, src): self.rs1 = f"={src}"

class RS2(IMPLICIT):
    def __init__(self, src): self.rs2 = f"={src}"

class IMM(IMPLICIT):
    def __init__(self, src): self.imm = f"={src}"


def measure(instructions):
    size = 0
    for ops in instructions:
        bits = 0
        count = 1
        display = []
        for op in ops:
            attrs = { 'rd', 'rs1', 'rs2', 'rs_imm', 'imm', 'suffix' }
            if isinstance(op[0], str):
                class Opcode:
                    def __init__(self, name):
                        self.__name__ = name
                        self.fmt = formats.get(name, f"{name:<7} {{rd}}, {{rs1}}, {{rs_imm}}")
                        self.bits = 0
                        self.count = 1
                op[0] = Opcode(op[0])
            opcode = op[0]
            args = op[1:]

            opbits = opcode.bits
            opcount = opcode.count
            fmt = opcode.fmt
            fields = {}
            for arg in args:
                for key in vars(arg).keys() & attrs:
                    fields[key] = getattr(arg, key)
            fields.setdefault('rs_imm', fields.get('rs2', None) or fields.get('imm', None))

            opbits += sum(map(lambda x: x.bits, args))
            opcount *= math.prod(map(lambda x: x.count, args))
            count *= opcount
            bits += opbits
            try:
                op = fmt.format(**fields)
            except Exception as err:
                print(f"Error formatting '{fmt}' with fields {fields}", err)
                raise err
            display.append(f"{opbits:2}: {op:<30}")

        def argname(arg):
            return arg if isinstance(arg, str) else arg.__name__
        def arglist(args):
            return map(argname, filter(lambda arg: arg.bits > 0, args))

        name = ",".join(map(lambda op: argname(op[0]), ops))
        name += " (" + " ".join([" ".join(arglist(op[1:])) for op in ops])+ ")"
        print(f"{size:#10x}{count:+#11x}: {"  ".join(display)}  ({bits:2} bits) {name:<40}")
        size += count

    print(f"total size: ({size:#x}),  bits: {(size - 1).bit_length()}")
    print()


rvc = [
    # Q0
    [[ "addi4spn",      rd_3, RS1("SP"), imm8 ]],
    [[ "fld/lq",        rd_3, rs1_3, imm5 ]],
    [[ "lw",            rd_3, rs1_3, imm5 ]],
    [[ "flw/ld",        rd_3, rs1_3, imm5 ]],
    [[ "fsd/sq",        rd_3, rs1_3, imm5 ]],
    [[ "sw",            rd_3, rs1_3, imm5 ]],
    [[ "fsw/sd",        rd_3, rs1_3, imm5 ]],

    # Q1
    [[ "nop" ]],
    [[ "addi",          rsd_nz, imm6 ]],
    [[ "jal",           RD("RA"), imm11 ]],
    [[ "addiw",         rsd_nz, imm6 ]],
    [[ "li",            rsd_nz, imm6 ]],
    [[ "addi16sp",      RD("SP"), RS1("SP"), imm6 ]],
    [[ "lui",           rsd_nz, imm6 ]],
    [[ "srli/64",       rsd_3, imm6 ]],
    [[ "srai/64",       rsd_3, imm6 ]],
    [[ "andi",          rsd_3, imm6 ]],
    [[ "sub",           rsd_3, rs2_3 ]],
    [[ "xor",           rsd_3, rs2_3 ]],
    [[ "or",            rsd_3, rs2_3 ]],
    [[ "and",           rsd_3, rs2_3 ]],
    [[ "subw",          rsd_3, rs2_3 ]],
    [[ "addw",          rsd_3, rs2_3 ]],
    [[ "j",             imm11 ]],
    [[ "beqz",          rs1_3, imm8 ]],
    [[ "bnez",          rs1_3, imm8 ]],

    # Q2
    [[ "slli",          rsd_nz, imm6 ]],
    [[ "fldsp/lqsp",    rd_nz, RS1("SP"), imm6 ]],
    [[ "lwsp",          rd_nz, RS1("SP"), imm6 ]],
    [[ "flwsp/ldsp",    rd_nz, RS1("SP"), imm6 ]],
    [[ "jr",            rs2 ]],
    [[ "mv",            rd_nz, rs2 ]],
    [[ "ebreak",        ]],
    [[ "jalr",          RD("RA"), rs2 ]],
    [[ "add",           rsd_nz, rs2 ]],
    [[ "fsdsp/sqsp",    rd, RS1("SP"), imm6 ]],
    [[ "swsp",          rd, RS1("SP"), imm6 ]],
    [[ "fswsp/sdsp",    rd, RS1("SP"), imm6 ]],
]

my_attempt = [
    [[ ari4, rsd, rs_imm            ],[ ari4, rsd, rs_imm ]],
    [[ ari4, RD("T6"), rs1, rs_imm  ],[ ari4, rd, RS1("T6"), rs_imm ]],
# share Rs2:
    [[ ari5, rsd, rs_imm            ],[ ari5x, rsd, RS2("rs2") ]],

# forward Rd to Rs2
    [[ ari5, rsd, rs_imm            ],[ ari5x, rsd, RS2("rd") ]],

    [[ ari4, rsd, rs_imm            ],[ "beqz", RS1("rd"), imm11 ]],
    [[ ari4, rsd, rs_imm            ],[ "bnez", RS1("rd"), imm11 ]],

    [[ cmp, RD("T6"), rs1, rs_imm   ],[ "beqz", RS1("T6"), imm11 ]],
    [[ cmp, RD("T6"), rs1, rs_imm   ],[ "bnez", RS1("T6"), imm11 ]],

    [[ ari4, rsd, rs_imm            ],[ "j", imm11 ]],
    [[ ari4, rsd, rs_imm            ],[ "jal", RD("RA"), imm11 ]],

    [[ ari5, rsd, rs_imm            ],[ "jr", rs2 ]],
    [[ ari5, rsd, rs_imm            ],[ "jalr", RD("RA"), rs2 ]],

    [[ "---", imm10, imm11 ]],

    [[ pair, rd, rs1, rs2           ],[ PAIR, rd, RS1("rs1"), RS2("rs2") ]],

    [[ ldst, rd, rs1, imm10,SHL     ],[ LDST, RD("Rd+1"), RS1("rs1"), IMM("Imm+1"),SHL ]],

    [[ ari3, rsd, rs_imm,SHL        ],[ ldst, rd, rs1, imm0 ]],
    [[ ldst, rd, rs1, imm0          ],[ ari3, rsd, rs_imm,SHL ]],
]

attempt2 = [
    [[ ari4, rsd, rs_imm            ],[ ari4, rsd, rs_imm ]],
    [[ ari4, RD("T6"), rs1, rs_imm  ],[ ari4, rd, RS1("T6"), rs_imm ]],
# share Rs2:
    [[ ari5, rsd, rs_imm            ],[ ari5x, rsd, RS2("rs2") ]],

# forward Rd to Rs2
    [[ ari5, rsd, rs_imm            ],[ ari5x, rsd, RS2("rd") ]],

    [[ ari4, rsd, rs_imm            ],[ "beqz", RS1("rd"), imm11 ]],
    [[ ari4, rsd, rs_imm            ],[ "bnez", RS1("rd"), imm11 ]],

    [[ cmp, RD("T6"), rs1, rs_imm   ],[ "beqz", RS1("T6"), imm11 ]],
    [[ cmp, RD("T6"), rs1, rs_imm   ],[ "bnez", RS1("T6"), imm11 ]],

    [[ ari4, rsd, rs_imm            ],[ "j", imm11 ]],
    [[ ari4, rsd, rs_imm            ],[ "jal", RD("RA"), imm11 ]],

    [[ ari5, rsd, rs_imm            ],[ "jr", rs2 ]],
    [[ ari5, rsd, rs_imm            ],[ "jalr", RD("RA"), rs2 ]],

    [[ ari5, rsd, rs_imm            ],[ "sw", RSD("rd"), RS1("SP"), imm8 ]],
    [[ ari5, rsd, rs_imm            ],[ "sd", RSD("rd"), RS1("SP"), imm8 ]],
    [[ "lw", rd, RS1("SP"), imm8    ],[ ari5, RSD("rd"), rs_imm ]],
    [[ "ld", rd, RS1("SP"), imm8    ],[ ari5, RSD("rd"), rs_imm ]],

    [[ "---", imm10, imm11 ]],

    [[ pair, rd, rs1, rs2           ],[ PAIR, rd, RS1("rs1"), RS2("rs2") ]],

    [[ ldst, rd, rs1, imm5,SHL      ],[ LDST, rd, RS1("rs1"), IMM("Imm+1"),SHL ]],

    [[ ari3, rsd, rs_imm,SHL        ],[ ldst, rd, rs1, imm0 ]],
    [[ ldst, rd, rs1, imm0          ],[ ari3, rsd, rs_imm,SHL ]],
]

measure(rvc)
measure(my_attempt)
measure(attempt2)
