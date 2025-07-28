import math

formats = {
    "nop": "nop",
    "---": "--reserved--",
#    "addi0":    "addi    {rd}, {rs1}, #{imm}",
#    "addi04spn":"addi    {rd}, SP, #{imm}*4",
#    "addi1":    "addi    {rd}, {rs1}, #{imm}-32",
#    "addi14spn":"addi    {rd}, SP, #{imm}*4+128",
#    "addi14spn":"addi    {rd}, SP, #{imm}*4+256",
#    "addi0w":   "addiw   {rd}, {rs1}, #{imm}",
#    "addi1w":   "addiw   {rd}, {rs1}, #{imm}-32",
#    "andi0":    "andi    {rd}, {rs1}, #{imm}",
#    "andi1":    "andi    {rd}, {rs1}, #{imm}+32",
#    "rsbi0":    "subi    {rd}, #{imm}, {rs1}",
#    "rsbi1":    "subi    {rd}, #{imm}-32, {rs1}",
#    "seqi0":    "slti    {rd}, {rs1}, #{rs_imm}",
#    "seqi1":    "slti    {rd}, {rs1}, #{rs_imm}",
#    "slli0":    "slli    {rd}, {rs1}, #{imm}",
#    "slli1":    "slli    {rd}, {rs1}, #{imm}-32",
#    "slti0":    "slti    {rd}, {rs1}, #{rs_imm}",
#    "slti0u":   "slti    {rd}, {rs1}, #{rs_imm}",
#    "slti1":    "slti    {rd}, {rs1}, #{rs_imm}+32",
#    "slti1u":   "slti    {rd}, {rs1}, #{rs_imm}+32",
#    "srai0":    "srai    {rd}, {rs1}, #{imm}",
#    "srai1":    "srai    {rd}, {rs1}, #{imm}-32",
#    "srli0":    "srli    {rd}, {rs1}, #{imm}",
#    "srli1":    "srli    {rd}, {rs1}, #{imm}-32",

    "bittesti0":"andi    {rd}, {rs1}, #1<<{rs_imm}",
    "bittesti1":"andi    {rd}, {rs1}, #1<<{rs_imm}+32",
    "beqz":     "beqz    {rs1}, +{imm}",
    "bnez":     "bnez    {rs1}, +{imm}",
    "j":        "j       +{imm}",
    "jr":       "jr      {rs2}",
    "jal":      "jal     {rd}, +{imm}",
    "jalr":     "jalr    {rd}, {rs2}",

    "ebreak":   "ebreak",
    "mv":       "mv      {rd}, {rs_imm}",

    "lb":       "lb      {rd}, {imm}({rs1})",
    "lh":       "lh      {rd}, {imm}*2({rs1})",
    "lw":       "lw      {rd}, {imm}*4({rs1})",
    "ld":       "ld      {rd}, {imm}*8({rs1})",
    "lq":       "lq      {rd}, {imm}*4({rs1})",
    "lbu":      "lbu     {rd}, {imm}({rs1})",
    "lhu":      "lhu     {rd}, {imm}*2({rs1})",
    "lwu/flw":  "lwu     {rd}, {imm}*4({rs1})",
    "ldu/fld":  "ldu     {rd}, {imm}*8({rs1})",
    "sb":       "sb      {rd}, {imm}({rs1})",
    "sh":       "sh      {rd}, {imm}*2({rs1})",
    "sw":       "sw      {rd}, {imm}*4({rs1})",
    "sd":       "sd      {rd}, {imm}*8({rs1})",
    "sq":       "sq      {rd}, {imm}*16({rs1})",
    "fsd":      "fsd     {rd}, {imm}*16({rs1})",
    "fsw":      "fsw     {rd}, {imm}*4({rs1})",
}

class Opcode:
    bits = 0
    count = 1
    def __init__(self, op):
        assert isinstance(op, str), f"{op=}, {type(op)=}"
        self.name = op
        self.choices = [ self.name ]


class OpSet(Opcode):
    def __init__(self, name, *ops, roundoff=True):
        self.name = name
        self.choices = ops
        self.count = len(self.choices)
        self.bits = (self.count - 1).bit_length()
        if roundoff: self.count = 1 << self.bits

    def __or__(self, other):
        return OpSet(
                "|".join((self.name, other.name)),
                *(set(self.choices) | set(other.choices)),
        )


def repack(*, rd=None, rs1=None, rsd=None, rs2=None, imm=None, rs_imm=None, shift=None):
    if rs_imm: assert not (rs2 or imm)
    if rsd: assert not (rd or rs1)
    assert not (rs2 and imm)

    d = {
        'rd': rd or rsd,
        'rs1': rs1 or rsd,
        'rs2': rs2,
        'imm': imm,
        'rs_imm': rs_imm or rs2 or imm,
        'shift': shift,
    }
    return { k:v for k, v in d.items() if v }


class Instruction:
    def __init__(self, operation, fmt=None, **kwargs):
        if isinstance(operation, str): operation = Opcode(operation)
        assert isinstance(operation, (Opcode, OpSet)), f"op={operation}, {repr(operation)}"
        self.operation = operation
        bits = operation.bits
        count = operation.count
        unique = set()
        for k, v in repack(**kwargs).items():
            setattr(self, k, v)
            unique.add(v)
        for v in unique:
            bits += v.bits
            count *= v.count
        self.bits = bits
        self.count = count
        if fmt:
            if hasattr(self, 'fmt'): print(f"replacing {self.fmt} with {fmt}")
            self.fmt = fmt
        elif not hasattr(self, 'fmt'):
            self.fmt = formats.get(self.operation.name, f"{self.operation.name:<7} {{rd}}, {{rs1}}, {{rs_imm}}")
    
    def __str__(self):
        return self.fmt.format(**vars(self))


class Register:
    def __init__(self, name, count = 32):
        self.name = name
        self.count = count
        self.bits = (self.count - 1).bit_length()

    def __str__(self):
        return self.name


class Register3(Register):
    def __init__(self, name):
        super().__init__(name, 8)

class Immediate:
    def __init__(self, size, name=None):
        self.name = name or f"imm{size}"
        self.bits = size
        self.count = 1 << size

    def __str__(self):
        return self.name


class RegImm(Register, Immediate):
    pass


class ari3(Instruction):
    operation = OpSet("ari3",
        "addi0",
        "addi1",
        "subi0",
        "subi1",
        "add",
        "sub",
        "and",
        "or",
    )
    fmt = "arith3  {rd}, {rs1}, {rs_imm}"
    def __init__(self, **kwargs):
        super().__init__(self.operation, **kwargs)

class ari4(Instruction):
    operation = OpSet("arith4",
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
    )
    fmt = "arith4  {rd}, {rs1}, {rs_imm}"
    def __init__(self, **kwargs):
        super().__init__(self.operation, **kwargs)

class ari5i(Instruction):
    operation = OpSet("arith5i",
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
    )
    fmt = "arith5i {rd}, {rs1}, {rs_imm}"
    def __init__(self, **kwargs):
        super().__init__(self.operation, **kwargs)

class ari5r(Instruction):
    operation = OpSet("arith5r",
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
    )
    fmt = "arith5r {rd}, {rs1}, {rs2}"
    def __init__(self, **kwargs):
        super().__init__(self.operation, **kwargs)

class ari5x(Instruction):
    operation = ari5i.operation  # TODO: fix this.
    fmt = "arith5x {rd}, {rs1}, {rs_imm}"
    def __init__(self, **kwargs):
        super().__init__(self.operation, **kwargs)

class ari5(Instruction):
    operation = ari5i.operation | ari5r.operation
    operation.name = "arith5"
    fmt = "arith5  {rd}, {rs1}, {rs_imm}"
    def __init__(self, **kwargs):
        super().__init__(self.operation, **kwargs)

class cmp(Instruction):
    operation = OpSet("cmpi",
        "slti0",
        "slti1",
        "slti0u",
        "slti1u",
        "seqi0",
        "seqi1",
        "bittesti0",
        "bittesti1",
    )
    fmt = "cmpi    {rd}, {rs1}, {imm}"
    def __init__(self, **kwargs):
        super().__init__(self.operation, **kwargs)

class ldst(Instruction):
    operation = OpSet("ldst",
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
    )
    fmt = "ldst    {rd}, {imm}({rs1})"
    def __init__(self, **kwargs):
        super().__init__(self.operation, **kwargs)


class pair(Instruction):
    opcode_pairs = [
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
    ]
    operation = OpSet("pair.a", *[p[0] for p in opcode_pairs], roundoff=False)
    print(f"{operation.count=}, {operation.bits=}, {len(operation.choices)=}, {operation.choices=}")
    fmt = "pair.a  {rd}, {rs1}, {rs2}"
    def __init__(self, **kwargs):
        super().__init__(self.operation, **kwargs)


class ImplicitInstruction(Instruction):
    def __init__(self, operation, **kwargs):
        assert isinstance(operation, str)
        super().__init__(operation, **kwargs)


class ImplicitRegister(Register):
    def __init__(self, name):
        super().__init__(name, 1)


class ImplicitImmediate(Immediate):
    def __init__(self, name):
        super().__init__(0, name)


class LDST(ImplicitInstruction):
    fmt = "=LdSt   {rd}, {imm}*k({rs1})"
    def __init__(self, **kwargs):
        super().__init__("=LdSt", **kwargs)


class PAIR(ImplicitInstruction):
    #operation = OpSet("pair.b", *[p[1] for p in pair.opcode_pairs], roundoff=False)
    fmt = "pair.b  {rd}, {rs1}, {rs2}"
    def __init__(self, **kwargs):
        super().__init__("=pair.b", **kwargs)


ZERO = ImplicitRegister("ZERO")
RA = ImplicitRegister("RA")
SP = ImplicitRegister("SP")
T6 = ImplicitRegister("T6")


rd = Register("rd")
rd_nz = Register("rd", 31)
rsd = Register("rsd")
rsd_nz = Register("rsd", 31)
rs1 = Register("rs1")
rs2 = Register("rs2")

rs_imm = RegImm("rs_imm")
rd_3 = Register3("rd")
rsd_3 = Register3("rsd")
rs1_3 = Register3("rs1")
rs2_3 = Register3("rs2")


SHL = ImplicitImmediate("<<k")

imm0 = Immediate(0, "0")
imm1 = Immediate(1)
imm2 = Immediate(2)
imm3 = Immediate(3)
imm4 = Immediate(4)
imm5 = Immediate(5)
imm6 = Immediate(6)
imm7 = Immediate(7)
imm8 = Immediate(8)
imm9 = Immediate(9)
imm10 = Immediate(10)
imm11 = Immediate(11)
imm12 = Immediate(12)
imm13 = Immediate(13)
imm14 = Immediate(14)
imm15 = Immediate(15)
imm16 = Immediate(16)
imm17 = Immediate(17)
imm18 = Immediate(18)
imm19 = Immediate(19)
imm20 = Immediate(20)
imm21 = Immediate(21)
imm22 = Immediate(22)
imm23 = Immediate(23)

class REUSE(ImplicitRegister):
    def __init__(self, src):
        super().__init__(f"={src}")


def dump(instructions):
    size = 0
    for ops in instructions:
        bits = 0
        count = 1
        display = []
        for op in ops:
            bits += op.bits
            count *= op.count
            display.append(f"{op.bits:2}: {str(op):<30}")

        print(f"{size:#10x}{count:+#11x}: {"  ".join(display)}  ({bits:2} bits)")
        size += count

    print(f"total size: {size:#x},  bits: {(size - 1).bit_length()}")
    print()


rvc = [
    # Q0
    ( Instruction("addi4spn",      rd=rd_3, rs1=SP, imm=imm8), ),
    ( Instruction("fld/lq",        rd=rd_3, rs1=rs1_3, imm=imm5), ),
    ( Instruction("lw",            rd=rd_3, rs1=rs1_3, imm=imm5), ),
    ( Instruction("flw/ld",        rd=rd_3, rs1=rs1_3, imm=imm5), ),
    ( Instruction("fsd/sq",        rd=rd_3, rs1=rs1_3, imm=imm5), ),
    ( Instruction("sw",            rd=rd_3, rs1=rs1_3, imm=imm5), ),
    ( Instruction("fsw/sd",        rd=rd_3, rs1=rs1_3, imm=imm5), ),

    # Q1
    ( Instruction("nop"), ),
    ( Instruction("addi",          rsd=rsd_nz, imm=imm6), ),
    ( Instruction("jal",           rd=RA, imm=imm11), ),
    ( Instruction("addiw",         rsd=rsd_nz, imm=imm6), ),
    ( Instruction("li",            rsd=rsd_nz, imm=imm6), ),
    ( Instruction("addi16sp",      rd=SP, rs1=SP, imm=imm6), ),
    ( Instruction("lui",           rsd=rsd_nz, imm=imm6), ),
    ( Instruction("srli/64",       rsd=rsd_3, imm=imm6), ),
    ( Instruction("srai/64",       rsd=rsd_3, imm=imm6), ),
    ( Instruction("andi",          rsd=rsd_3, imm=imm6), ),
    ( Instruction("sub",           rsd=rsd_3, rs2=rs2_3), ),
    ( Instruction("xor",           rsd=rsd_3, rs2=rs2_3), ),
    ( Instruction("or",            rsd=rsd_3, rs2=rs2_3), ),
    ( Instruction("and",           rsd=rsd_3, rs2=rs2_3), ),
    ( Instruction("subw",          rsd=rsd_3, rs2=rs2_3), ),
    ( Instruction("addw",          rsd=rsd_3, rs2=rs2_3), ),
    ( Instruction("j",             imm=imm11), ),
    ( Instruction("beqz",          rs1=rs1_3, imm=imm8), ),
    ( Instruction("bnez",          rs1=rs1_3, imm=imm8), ),

    # Q2
    ( Instruction("slli",          rsd=rsd_nz, imm=imm6), ),
    ( Instruction("fldsp/lqsp",    rd=rd_nz, rs1=SP, imm=imm6), ),
    ( Instruction("lwsp",          rd=rd_nz, rs1=SP, imm=imm6), ),
    ( Instruction("flwsp/ldsp",    rd=rd_nz, rs1=SP, imm=imm6), ),
    ( Instruction("jr",            rs2=rs2), ),
    ( Instruction("mv",            rd=rd_nz, rs2=rs2), ),
    ( Instruction("ebreak",       ), ),
    ( Instruction("jalr",          rd=RA, rs2=rs2), ),
    ( Instruction("add",           rsd=rsd_nz, rs2=rs2), ),
    ( Instruction("fsdsp/sqsp",    rd=rd, rs1=SP, imm=imm6), ),
    ( Instruction("swsp",          rd=rd, rs1=SP, imm=imm6), ),
    ( Instruction("fswsp/sdsp",    rd=rd, rs1=SP, imm=imm6), ),
]

my_attempt = [
    ( ari4(rsd=rsd, rs_imm=rs_imm),             ari4(rsd=rsd, rs_imm=rs_imm), ),
    ( ari4(rd=T6, rs1=rs1, rs_imm=rs_imm),      ari4(rd=rd, rs1=T6, rs_imm=rs_imm), ),
# share Rs2:
    ( ari5(rsd=rsd, rs_imm=rs_imm),             ari5x(rsd=rsd, rs2=REUSE("rs2")), ),

# forward Rd to Rs2
    ( ari5(rsd=rsd, rs_imm=rs_imm),             ari5x(rsd=rsd, rs2=REUSE("rd")), ),

    ( ari4(rsd=rsd, rs_imm=rs_imm),             Instruction("beqz", rs1=REUSE("rd"), imm=imm11), ),
    ( ari4(rsd=rsd, rs_imm=rs_imm),             Instruction("bnez", rs1=REUSE("rd"), imm=imm11), ),

    ( cmp(rd=T6, rs1=rs1, imm=imm5),            Instruction("beqz", rs1=T6, imm=imm11), ),
    ( cmp(rd=T6, rs1=rs1, imm=imm5),            Instruction("bnez", rs1=T6, imm=imm11), ),

    ( ari4(rsd=rsd, rs_imm=rs_imm),             Instruction("j", imm=imm11), ),
    ( ari4(rsd=rsd, rs_imm=rs_imm),             Instruction("jal", rd=RA, imm=imm11), ),

    ( ari5(rsd=rsd, rs_imm=rs_imm),             Instruction("jr", rs2=rs2), ),
    ( ari5(rsd=rsd, rs_imm=rs_imm),             Instruction("jalr", rd=RA, rs2=rs2), ),

    ( Instruction("---", imm=imm21), ),

    ( pair(rd=rd, rs1=rs1, rs2=rs2),            PAIR(rd=rd, rs1=REUSE("rs1"), rs2=REUSE("rs2")), ),

    ( ldst(rd=rd, rs1=rs1, imm=imm10, shift=SHL), LDST(rd=REUSE("Rd+1"), rs1=REUSE("rs1"), imm=REUSE("Imm+1"),shift=SHL), ),

    ( ari3(rsd=rsd, rs_imm=rs_imm,shift=SHL),   ldst(rd=rd, rs1=rs1, imm=imm0), ),
    ( ldst(rd=rd, rs1=rs1, imm=imm0),           ari3(rsd=rsd, rs_imm=rs_imm,shift=SHL), ),
]

attempt2 = [
    ( ari4(rsd=rsd, rs_imm=rs_imm),             ari4(rsd=rsd, rs_imm=rs_imm), ),
    ( ari4(rd=T6, rs1=rs1, rs_imm=rs_imm),      ari4(rd=rd, rs1=T6, rs_imm=rs_imm), ),
# share Rs2:
    ( ari5(rsd=rsd, rs_imm=rs_imm),             ari5x(rsd=rsd, rs2=REUSE("rs2")), ),

# forward Rd to Rs2
    ( ari5(rsd=rsd, rs_imm=rs_imm),             ari5x(rsd=rsd, rs2=REUSE("rd")), ),

    ( ari4(rsd=rsd, rs_imm=rs_imm),             Instruction("beqz", rs1=REUSE("rd"), imm=imm11), ),
    ( ari4(rsd=rsd, rs_imm=rs_imm),             Instruction("bnez", rs1=REUSE("rd"), imm=imm11), ),

    ( cmp(rd=T6, rs1=rs1, imm=imm5),            Instruction("beqz", rs1=T6, imm=imm11), ),
    ( cmp(rd=T6, rs1=rs1, imm=imm5),            Instruction("bnez", rs1=T6, imm=imm11), ),

    ( ari4(rsd=rsd, rs_imm=rs_imm),             Instruction("j", imm=imm11), ),
    ( ari4(rsd=rsd, rs_imm=rs_imm),             Instruction("jal", rd=RA, imm=imm11), ),

    ( ari5(rsd=rsd, rs_imm=rs_imm),             Instruction("jr", rs2=rs2), ),
    ( ari5(rsd=rsd, rs_imm=rs_imm),             Instruction("jalr", rd=RA, rs2=rs2), ),

    ( ari5(rsd=rsd, rs_imm=rs_imm),             Instruction("sw", rd=REUSE("rd"), rs1=SP, imm=imm8), ),
    ( ari5(rsd=rsd, rs_imm=rs_imm),             Instruction("sd", rd=REUSE("rd"), rs1=SP, imm=imm8), ),
    ( Instruction("lw", rd=rd, rs1=SP, imm=imm8), ari5(rsd=REUSE("rd"), rs_imm=rs_imm), ),
    ( Instruction("ld", rd=rd, rs1=SP, imm=imm8), ari5(rsd=REUSE("rd"), rs_imm=rs_imm), ),

    ( Instruction("---", imm=imm21), ),

    ( pair(rd=rd, rs1=rs1, rs2=rs2),            PAIR(rd=rd, rs1=REUSE("rs1"), rs2=REUSE("rs2")), ),

    ( ldst(rd=rd, rs1=rs1, imm=imm5,shift=SHL), LDST(rd=rd, rs1=REUSE("rs1"), imm=REUSE("Imm+1"), shift=SHL), ),

    ( ari3(rsd=rsd, rs_imm=rs_imm,shift=SHL),   ldst(rd=rd, rs1=rs1, imm=imm0), ),
    ( ldst(rd=rd, rs1=rs1, imm=imm0),           ari3(rsd=rsd, rs_imm=rs_imm,shift=SHL), ),
]

dump(rvc)
dump(my_attempt)
dump(attempt2)
