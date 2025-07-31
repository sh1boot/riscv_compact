import math
import re

class Opcode:
    bits = 0
    count = 1
    def __init__(self, op, aliases=()):
        assert isinstance(op, str), f"{op=}, {type(op)=}"
        self.name = op
        self.re = self.name if not aliases else f"({'|'.join([op]+aliases)})"
        self.choices = [ self.name ]
    def __str__(self):
        return self.name
    def __format__(self, spec):
        return self.__str__().__format__(spec)


class OpSet(Opcode):
    def __init__(self, name, *ops, roundoff=True, aliases=()):
        self.name = name
        self.re = f"({'|'.join(ops+aliases)})"
        self.choices = ops
        self.count = len(self.choices)
        self.bits = (self.count - 1).bit_length()
        if roundoff: self.count = 1 << self.bits

    def __or__(self, other):
        return OpSet(
                "|".join((self.name, other.name)),
                *(set(self.choices) | set(other.choices)),
        )


def repack(*, rd=None, rs1=None, rsd=None, rs2=None, imm=None, rs_imm=None, shift=None, **kwargs):
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
    kwargs.update({ k:v for k, v in d.items() if v })
    return kwargs


class Instruction:
    def __init__(self, opcode, **kwargs):
        if isinstance(opcode, str): opcode = Opcode(opcode)
        assert isinstance(opcode, (Opcode, OpSet)), f"op={opcode}, {repr(opcode)}"
        self.name = f"Instruction({opcode.name})"
        self.opcode = opcode
        bits = opcode.bits
        count = opcode.count
        arglist = []
        unique = set()
        #for k, v in repack(**kwargs).items():
        for k, v in kwargs.items():
            if k == 'rsd':
                arglist.append('rd')
                arglist.append('rs1')
                setattr(self, 'rd', v)
                setattr(self, 'rs1', v)
            else:
                arglist.append(k)
                setattr(self, k, v)
            unique.add(v)
        for v in unique:
            bits += v.bits
            count *= v.count
        self.bits = bits
        self.count = count

        if arglist:
            comma = '}, {'
            auto_fmt = f"{self.opcode.name:<7} {{{comma.join(arglist)}}}"
            re_comma = r'\s*,\s*'
            re_list = [ f"(?P<{k}>{getattr(self, k).re})" for k in arglist ]
            auto_re = f"\\b(?P<opcode>{self.opcode.re})\\s+{re_comma.join(re_list)}"
        else:
            auto_fmt = f"{self.opcode.name}"
            auto_re = f"{self.opcode.name}"

        if not hasattr(self, 'fmt'):
            self.fmt = formats.get(self.opcode.name, auto_fmt)

        if not hasattr(self, 're'):
            self.re = regexps.get(self.opcode.name, auto_re)
        self.re_nocapture = re.sub(r'\(\?P<\w+>', '(', self.re)

        self.re_prog = re.compile(self.re, re.ASCII)

    def __str__(self):
        fmt_args = repack(**vars(self))
        try:
            return self.fmt.format(**fmt_args)
        except Exception as e:
            print(f"problem formatting: {self.fmt}, from: {fmt_args.keys()}")
            raise e
    def __format__(self, spec):
        return self.__str__().__format__(spec)

    def search(self, *args, **kwargs):
        if not (match := self.re_prog.search(*args, **kwargs)):
            return None
        for k, v in match.groupdict().items():
            validate = getattr(getattr(self, k, None), 'validate', None)
            if validate and not validate(v):
                return None
        return match
    def cook_re(self, match):
        fmt_args = repack(**match.groupdict())
        fmt_re = self.re_nocapture
        try:
            return re.compile(fmt_re.format(**fmt_args), re.ASCII)
        except KeyError as e:
            # I think this is probably OK, but might revisit.
            print(f"Trouble with '{fmt_re} looking for {e} in {fmt_args}")
            pass
        return None


class Register:
    re = r'\b(x[12]?\d|x3[01]|zero|ra|[sgtf]p|t[0-6]|s\d|s1[01]|a[0-7])\b'
    def __init__(self, name, count = 32):
        self.name = name
        self.count = count
        self.bits = (self.count - 1).bit_length()

    def __str__(self):
        return self.name
    def __format__(self, spec):
        return self.__str__().__format__(spec)


class Register3(Register):
    re = r'\b(x[89]|x1[0-5]|fp|s[01]|a[0-5])\b'
    def __init__(self, name):
        super().__init__(name, 8)

class Immediate:
    re = r'[-+]?\d+\b'
    def __init__(self, size, name=None):
        self.name = name or f"imm{size}"
        self.bits = size
        self.count = 1 << size

    def __str__(self):
        return self.name
    def __format__(self, spec):
        return self.__str__().__format__(spec)


class RegImm(Register, Immediate):
    re = f'({Register.re}|{Immediate.re})'
    def __init__(self, name, reg_name, imm_name='imm', count=32):
        super().__init__(name, count)
        re = fr'((?P<{reg_name}>{Register.re})|(?P<{imm_name}>{Immediate.re}))'

class ari3(Instruction):
    opcode = OpSet("arith3",
        "addi",         # +0
        "addi",         # +32
        "subi",         # +0
        "subi",         # +32
        "add",
        "sub",
        "and",
        "or",
    )
    def __init__(self, **kwargs):
        super().__init__(self.opcode, **kwargs)

class ari4(Instruction):
    opcode = OpSet("arith4",
        "addi",         # +0
        "addi",         # -32
        "addiw",        # +0
        "addiw ",       # -32
        "addi4spn",     # +0
        "addi4spn",     # -32
        "andi",         # +0
        "andi",         # -32
        "add",
        "addw",
        "sub",
        "subw",
        "and",
        "bic",
        "or",
        "xor",
    )
    def __init__(self, **kwargs):
        super().__init__(self.opcode, **kwargs)

class ari5i(Instruction):
    opcode = OpSet("arith5i",
        "addi",         # +0
        "addi",         # -32
        "addiw",        # +0
        "addiw",        # -32
        "addi4spn",     # +0
        "addi4spn",     # -32
        "andi",         # +0
        "andi",         # -32
        "slli",         # +0
        "slli",         # +32
        "srli",         # +0
        "srli",         # +32
        "srai",         # +0
        "srai",         # +32
        "rsbi",         # +0
        "rsbi",         # +32
    )
    def __init__(self, **kwargs):
        super().__init__(self.opcode, **kwargs)

class ari5r(Instruction):
    opcode = OpSet("arith5r",
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
    )
    def __init__(self, **kwargs):
        super().__init__(self.opcode, **kwargs)

class ari5(Instruction):
    opcode = ari5i.opcode | ari5r.opcode
    opcode.name = "arith5"
    def __init__(self, **kwargs):
        super().__init__(self.opcode, **kwargs)

class cmp(Instruction):
    opcode = OpSet("cmpi",
        "slti",         # +0
        "slti",         # -32
        "sltiu",        # +0
        "sltiu",        # +32
        "seqi",         # +0
        "seqi",         # -32
        "bittesti",     # +0
        "bittesti",     # +32
    )
    def __init__(self, **kwargs):
        super().__init__(self.opcode, **kwargs)

class ldst(Instruction):
    opcode = OpSet("ldst",
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
        "lwu",      # or "flw"
        "fld",      # or "ldu"
        "fsw",
        "fsd",
    )
    fmt = "{opcode:<7} {rd}, {imm}({rs1})"
    re = fr"\b(?P<opcode>{opcode.re})\s+(?P<rd>{Register.re})\s*,\s*(?P<imm>{Immediate.re})\((?P<rs1>{Register.re})\)"
    def __init__(self, **kwargs):
        super().__init__(self.opcode, **kwargs)


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
        ("undef_paira", "undef_pairb"),
        ("undef_paira", "undef_pairb"),
    ]
    opcode = OpSet("pair.a", *[p[0] for p in opcode_pairs], roundoff=False)
    def __init__(self, **kwargs):
        super().__init__(self.opcode, **kwargs)


class Reserved(Instruction):
    opcode = Opcode("--reserved--")
    def __init__(self, bits, message=None):
        super().__init__(self.opcode)
        self.bits = bits
        self.count = 1 << bits
        self.fmt = message or "--reserved--"


class ImplicitInstruction(Instruction):
    def __init__(self, opcode, **kwargs):
        assert isinstance(opcode, str)
        super().__init__(opcode, **kwargs)


class ImplicitRegister(Register):
    def __init__(self, name):
        super().__init__(name, 1)
        self.re = name


class ImplicitImmediate(Immediate):
    def __init__(self, name):
        super().__init__(0, name)
        self.re = name


class LDST(ImplicitInstruction):
    arglist = ( 'opcode', 'rd', 'imm', 'rs1' )
    fmt = ldst.fmt
    re = ldst.re
    def __init__(self, **kwargs):
        super().__init__("=LdSt", **kwargs)


class PAIR(ImplicitInstruction):
    #opcode = OpSet("pair.b", *[p[1] for p in pair.opcode_pairs], roundoff=False)
    def __init__(self, **kwargs):
        super().__init__("=Pair.B", **kwargs)


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

rs_imm = RegImm("rs_imm", "rs2")
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
        super().__init__(f"{{{src}}}")

def capture(name, value): return fr'(?P<{name}>{value})'
def cap_obj(obj): return capture(obj.name, obj.re)
def cap_opc(opc): return capture('opcode', opc)
def cap_imm(): return capture('imm', r'[-+]?\d+')

LDST_regex = fr"{cap_obj(ldst.opcode)}\s+{cap_obj(rd)}\s*,\s*{cap_imm()}\({cap_obj(rs1)}\)"
LDSTSP_regex = fr"{cap_obj(ldst.opcode)}\s+{cap_obj(rd)}\s*,\s*{cap_imm()}\(sp\)"

LDST_format = "{opcode:<7} {rd}, {imm}({rs1})"
LDSTSP_format = "{opcode:<7} {rd}, {imm}(sp)"
formats = {
    "mv":       "{opcode:<7} {rd}, {rs_imm}",

    "lb":       LDST_format,
    "lh":       LDST_format,
    "lw":       LDST_format,
    "ld":       LDST_format,
    "lq":       LDST_format,
    "lbu":      LDST_format,
    "lhu":      LDST_format,
    "lwu/flw":  LDST_format,
    "ldu/fld":  LDST_format,
    "lwu":      LDST_format,
    "ldu":      LDST_format,
    "flw":      LDST_format,
    "fld":      LDST_format,
    "sb":       LDST_format,
    "sh":       LDST_format,
    "sw":       LDST_format,
    "sd":       LDST_format,
    "sq":       LDST_format,
    "fsd":      LDST_format,
    "fsw":      LDST_format,


    "fsdsp/sqsp":LDSTSP_format,
    "swsp":     LDSTSP_format,
    "fswsp/sdsp":LDSTSP_format,
    "fld/lq":   LDST_format,
    "flw/ld":   LDST_format,
    "fsd/sq":   LDST_format,
    "fsw/sd":   LDST_format,
}

regexps = {
    "mv":       fr"{cap_opc('mv')}\s+{cap_obj(rsd)}\s*,\s*{cap_obj(rs_imm)}",

    "lb":       LDST_regex,
    "lh":       LDST_regex,
    "lw":       LDST_regex,
    "ld":       LDST_regex,
    "lq":       LDST_regex,
    "lbu":      LDST_regex,
    "lhu":      LDST_regex,
    "lwu/flw":  LDST_regex,
    "ldu/fld":  LDST_regex,
    "lwu":      LDST_regex,
    "ldu":      LDST_regex,
    "flw":      LDST_regex,
    "fld":      LDST_regex,
    "sb":       LDST_regex,
    "sh":       LDST_regex,
    "sw":       LDST_regex,
    "sd":       LDST_regex,
    "sq":       LDST_regex,
    "fsd":      LDST_regex,
    "fsw":      LDST_regex,


    "fsdsp/sqsp":LDSTSP_regex,
    "swsp":     LDSTSP_regex,
    "fswsp/sdsp":LDSTSP_regex,
    "fld/lq":   LDST_regex,
    "flw/ld":   LDST_regex,
    "fsd/sq":   LDST_regex,
    "fsw/sd":   LDST_regex,
}




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
    ( ari5i(rsd=rsd, imm=imm5),                 ari5i(rsd=rsd, imm=REUSE("imm")), ),
    ( ari5r(rsd=rsd, rs2=rs2),                  ari5r(rsd=rsd, rs2=REUSE("rs2")), ),

# forward Rd to Rs2
    ( ari5i(rsd=rsd, imm=imm5),                 ari5r(rsd=rsd, rs2=REUSE("rd")), ),  # WUT?
    ( ari5r(rsd=rsd, rs2=rs2),                  ari5r(rsd=rsd, rs2=REUSE("rd")), ),

    ( ari4(rsd=rsd, rs_imm=rs_imm),             Instruction("beqz", rs1=REUSE("rd"), imm=imm11), ),
    ( ari4(rsd=rsd, rs_imm=rs_imm),             Instruction("bnez", rs1=REUSE("rd"), imm=imm11), ),

    ( cmp(rd=T6, rs1=rs1, imm=imm5),            Instruction("beqz", rs1=T6, imm=imm11), ),
    ( cmp(rd=T6, rs1=rs1, imm=imm5),            Instruction("bnez", rs1=T6, imm=imm11), ),

    ( ari4(rsd=rsd, rs_imm=rs_imm),             Instruction("j", imm=imm11), ),
    ( ari4(rsd=rsd, rs_imm=rs_imm),             Instruction("jal", rd=RA, imm=imm11), ),

    ( ari5(rsd=rsd, rs_imm=rs_imm),             Instruction("jr", rs2=rs2), ),
    ( ari5(rsd=rsd, rs_imm=rs_imm),             Instruction("jalr", rd=RA, rs2=rs2), ),

    ( Reserved(21), ),

    ( pair(rd=rd, rs1=rs1, rs2=rs2),            PAIR(rd=rd, rs1=REUSE("rs1"), rs2=REUSE("rs2")), ),

    ( ldst(rd=rd, rs1=rs1, imm=imm10, shift=SHL), LDST(rd=REUSE("Rd+1"), rs1=REUSE("rs1"), imm=REUSE("Imm+1"),shift=SHL), ),

    ( ari3(rsd=rsd, rs_imm=rs_imm,shift=SHL),   ldst(rd=rd, rs1=rs1, imm=imm0), ),
    ( ldst(rd=rd, rs1=rs1, imm=imm0),           ari3(rsd=rsd, rs_imm=rs_imm,shift=SHL), ),
]

attempt2 = [
    ( ari4(rsd=rsd, rs_imm=rs_imm),             ari4(rsd=rsd, rs_imm=rs_imm), ),
    ( ari4(rd=T6, rs1=rs1, rs_imm=rs_imm),      ari4(rd=rd, rs1=T6, rs_imm=rs_imm), ),
# share Rs2:
    ( ari5i(rsd=rsd, imm=imm5),                 ari5i(rsd=rsd, imm=REUSE("imm")), ),
    ( ari5r(rsd=rsd, rs2=rs2),                  ari5r(rsd=rsd, rs2=REUSE("rs2")), ),

# forward Rd to Rs2
    ( ari5i(rsd=rsd, imm=imm5),                 ari5r(rsd=rsd, rs2=REUSE("rd")), ),  # WUT?
    ( ari5r(rsd=rsd, rs2=rs2),                  ari5r(rsd=rsd, rs2=REUSE("rd")), ),

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

    ( Reserved(21), ),

    ( pair(rd=rd, rs1=rs1, rs2=rs2),            PAIR(rd=rd, rs1=REUSE("rs1"), rs2=REUSE("rs2")), ),

    ( ldst(rd=rd, rs1=rs1, imm=imm5,shift=SHL), LDST(rd=rd, rs1=REUSE("rs1"), imm=REUSE("imm"), shift=SHL), ),

    ( ari3(rsd=rsd, rs_imm=rs_imm,shift=SHL),   ldst(rd=rd, rs1=rs1, imm=imm0), ),
    ( ldst(rd=rd, rs1=rs1, imm=imm0),           ari3(rsd=rsd, rs_imm=rs_imm,shift=SHL), ),
]


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


def first_line(instructions, line):
    return filter(None, ( row[1].cook_re(match) for row in instructions if (match := row[0].search(line)) ))


def next_line(patterns, line):
    return next((m for r in patterns if (m := r.search(line))), None)


def try_pair(instructions, line0, line1):
    patterns = tuple(first_line(instructions, line0))
    print("input0:", line0)
    for pat in patterns:
        print(pat)

    print("\ninput1:", line1)
    match = next_line(patterns, line1)
    print(match)


def compress(instructions, filename):
    saved = 0
    total = 0
    with open(filename, "rt") as f:
        patterns = []
        for line in f:
            line = line.strip()
            if not line.startswith('0x'):
                print('#', line)
                continue

            match = next_line(patterns, line)
            if match:
                print("^^", line)
                patterns = []
                saved += 1
            else:
                failed_patterns = patterns
                patterns = tuple(first_line(instructions, line))
                print(f"{len(patterns):2} {line}")
                #if failed_patterns:
                #    print(f"  (all failed: {failed_patterns})")
            total += 1

        print(f"{saved=}, {total=}")

dump(rvc)
dump(my_attempt)
dump(attempt2)
try_pair(attempt2, " slli a4,a1,48", " srli a4,a4,48")

compress(attempt2, "qemu-lite.txt")
