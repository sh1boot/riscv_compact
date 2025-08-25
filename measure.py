import copy
from enum import Enum
import re

DEFAULT_IMM_SHIFT = 3

LDST_regex = r"\b{opcode}\s+{rd},\s*{imm}[(]{rs1}[)]"
LDSTSP_regex = r"\b{opcode}\s+{rd},\s*{imm}[(]sp[)]"
REG_regex = r'zero|[astx][0123]?\d|[sgtf]p|ra'
IMM_regex = r'[-+]?\d+'

unaliases = {
    fr"\bmv\s+(?P<rd>{REG_regex}),\s*(?P<rs2>{REG_regex})":   r"add \g<rd>,zero,\g<rs2>",
    fr"\bmv\s+(?P<rd>{REG_regex}),\s*(?P<imm>{IMM_regex})":   r"addi \g<rd>,zero,\g<imm>",
    r"\bret\b":   "jr      ra",
}

LDST_format = "{opcode:<7} {rd},{imm}({rs1})"
LDSTSP_format = "{opcode:<7} {rd},{imm}(sp)"
formats = {
    "mv":       "{opcode:<7} {rd},{rs_imm}",

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
    "lwsp":     LDSTSP_format,
    "swsp":     LDSTSP_format,
    "fswsp/sdsp":LDSTSP_format,
    "fld/lq":   LDST_format,
    "flw/ld":   LDST_format,
    "fsd/sq":   LDST_format,
    "fsw/sd":   LDST_format,
}

regexps = {
    "mv":       r"{opcode}\s+{rd},{rs_imm}",

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
    "lwsp":     LDSTSP_regex,
    "swsp":     LDSTSP_regex,
    "fswsp/sdsp":LDSTSP_regex,
    "fld/lq":   LDST_regex,
    "flw/ld":   LDST_regex,
    "fsd/sq":   LDST_regex,
    "fsw/sd":   LDST_regex,
}


wordsizes = {
    "sh1add": 1,
    "sh2add": 2,
    "sh3add": 3,
    "sh4add": 4,
    "sh1sub": 1,
    "sh2sub": 2,
    "sh3sub": 3,
    "sh4sub": 4,
    "sh1addi": 1,
    "sh2addi": 2,
    "sh3addi": 3,
    "sh4addi": 4,
    "sh1subi": 1,
    "sh2subi": 2,
    "sh3subi": 3,
    "sh4subi": 4,

    "lh": 1,
    "lhu": 1,
    "sh": 1,
    "lw": 2,
    "lwu": 2,
    "sw": 2,
    "flw": 2,
    "fsw": 2,
    "lwu/flw": 2,
    "lwsp": 2,
    "swsp": 2,

    "ld": 3,
    "ldu": 3,
    "sd": 3,
    "fld": 3,
    "fsd": 3,
    "ldu/fld": 3,
    "ldsp": 3,
    "sdsp": 3,

    "lq": 4,
    "sq": 4,

    "addi4spn": 2,
}

unsigned_immediates = {
    "addi4spn",
    "andi",
    "slli",
    "srli",
    "srai",
    "rsbi",
    "sltiu",
    "bittesti",
    "sdsp",
    "sqsp",
    "lwsp",
    "swsp",
    "flwsp",
    "fldsp",
    "fswsp",
    "fsdsp",
}


class ChoiceOf:
    bits = 0
    count = 1
    name = "unnamed"
    def __init__(self, name, *, choices=(), roundoff=True):
        self.name = name
        self.choices = choices or (name,)
        self.count = len(self.choices)
        self.bits = (self.count - 1).bit_length()
        if roundoff:
            self.count = 1 << self.bits

    def __str__(self):
        if getattr(self, 'shift', 0) > 0:
            if len(self.choices) > 1:
                return f"{self.name}*{1<<self.shift}:{(len(self.choices) - 1).bit_length()}"
            return f"{self.name}*{1<<self.shift}"
        if len(self.choices) > 1:
            return f"{self.name}:{(len(self.choices) - 1).bit_length()}"
        return self.name

    def __format__(self, spec):
        return self.__str__().__format__(spec)

    def dup(self, *args, **kwargs):
        retval = copy.copy(self)
        retval.__init__(*args, **kwargs)
        return retval

    def __add__(self, other):
        assert isinstance(other, int)
        index = self.choices.index(self.name) + other
        if 0 <= index and index < len(self.choices):
            return self.dup(self.choices[index])
        return None

    def __sub__(self, other):
        assert isinstance(other, int)
        return self.__add__(-other)


class Opcode(ChoiceOf):
    def __init__(self, name, *, choices=(), roundoff=True, hints={}):
        super().__init__(name, choices=choices, roundoff=roundoff)

    def __add__(self, other):
        return Opcode(
                name="+".join((self.name, other.name)),
                choices=tuple(tuple(self.choices) + tuple(other.choices))
        )

    def parse(self, string, hints={}):
        if string not in self.choices:
            raise ValueError(f"'{string}' not in {self.name}")
        return Opcode(string)


class RegName(Enum):
     zero = 0
     ra  = 1
     sp  = 2
     gp  = 3
     tp  = 4
     t0  = 5
     t1  = 6
     t2  = 7
     s0  = 8
     s1  = 9
     a0  = 10
     a1  = 11
     a2  = 12
     a3  = 13
     a4  = 14
     a5  = 15
     a6  = 16
     a7  = 17
     s2  = 18
     s3  = 19
     s4  = 20
     s5  = 21
     s6  = 22
     s7  = 23
     s8  = 24
     s9  = 25
     s10 = 26
     s11 = 27
     t3  = 28
     t4  = 29
     t5  = 30
     t6  = 31

     fp  = 8

     x0  = 0
     x1  = 1
     x2  = 2
     x3  = 3
     x4  = 4
     x5  = 5
     x6  = 6
     x7  = 7
     x8  = 8
     x9  = 9
     x10 = 10
     x11 = 11
     x12 = 12
     x13 = 13
     x14 = 14
     x15 = 15
     x16 = 16
     x17 = 17
     x18 = 18
     x19 = 19
     x20 = 20
     x21 = 21
     x22 = 22
     x23 = 23
     x24 = 24
     x25 = 25
     x26 = 26
     x27 = 27
     x28 = 28
     x29 = 29
     x30 = 30
     x31 = 31

class Register(ChoiceOf):
    choices = tuple(RegName)

    def __init__(self, name, *, choices=None, nonzero=False, **kwargs):
        choices = choices or (self.choices if not nonzero else self.choices[1:32])
        super().__init__(name, choices=choices, roundoff=False, **kwargs)

    def parse(self, value, hints={}):
        if isinstance(value, str):
            value = RegName[value]
        elif isinstance(value, int):
            value = RegName(value)
        assert isinstance(value, RegName)
        if value not in self.choices:
            raise ValueError(f"{value} not one of {self.choices=}")
        return SpecificRegister(value)


class SpecificRegister(Register):
    def __init__(self, reg):
        assert isinstance(reg, RegName)
        super().__init__(reg.name, choices=(reg,))


class Register4(Register):
    choices = tuple(map(RegName, range(16)))
    def __init__(self, name):
        super().__init__(name, choices=Register4.choices)


class Register3(Register):
    choices = tuple(map(RegName, range(8,16)))
    def __init__(self, name):
        super().__init__(name, choices=Register3.choices)


class Register4(Register):
    choices = tuple(map(RegName, range(0,16)))
    def __init__(self, name, **kwargs):
        super().__init__(name, choices=Register4.choices, **kwargs)

def bits_to_range(bits, signed=False, shift=0):
    count = 1 << bits
    start = -(count // 2) if signed else 0
    k = 1 << shift
    return range(start * k, (start + count) * k, k)

class Immediate:
    shift = 0

    def __init__(self, size=0, name=None, choices=(), hints={}):
        assert bool(size > 0) ^ bool(choices), f"{size=}, {choices=}"
        self.name = name or f"imm{size}"
        self.bits = size
        self.count = 1 << size
        self.shift = hints.get('shift', DEFAULT_IMM_SHIFT)
        self.signed = hints.get('signed', True)
        if choices:
            self.choices = choices
        else:
            self.choices = bits_to_range(self.bits, signed=self.signed, shift=self.shift)

    def __str__(self):
        return self.name

    def __format__(self, spec):
        return self.__str__().__format__(spec)

    def __add__(self, other):
        assert isinstance(other, int)
        assert self.choices
        other <<= self.shift
        choices = tuple(x + other for x in self.choices)
        return self.dup(name=f"{self.name}+{other}", choices=choices)

    def __sub__(self, other):
        assert isinstance(other, int)
        return self.__add__(-other)

    def dup(self, *args, **kwargs):
        retval = copy.copy(self)
        retval.__init__(*args, **kwargs)
        return retval

    def parse(self, string, hints={}):
        value = int(string)
        bits = self.bits + hints.get('extra_bits', 0)
        shift = hints.get('shift', self.shift)
        signed = hints.get('signed', self.signed)
        choices = bits_to_range(self.bits, signed=self.signed, shift=self.shift)
        if value not in choices:
            raise ValueError(f"{value=} not in {choices}")
        # TODO: ensure shift gets into result:
        return SpecificImmediate(name=string, value=value, hints=hints)


class SpecificImmediate(Immediate):
    def __init__(self, name, value=None, choices=(), hints={}):
        assert (value is not None) ^ bool(choices), f"{value=}, {choices=}"
        choices = choices or (value,)
        super().__init__(size=0, name=name, choices=choices, hints=hints)
        self.value = value


class RegImm(Register, Immediate):
    def __init__(self, name, reg_name, imm_name='imm', choices=(), hints={}):
        super().__init__(name, choices=choices)
        self.shift = hints.get('shift', self.shift)
        self.signed = hints.get('signed', True)
        print(f"Made {self} with {self.shift=}")

    def parse(self, string, hints={}):
        try:
            return Register.parse(self, string, hints)
        except (ValueError,KeyError):
            pass
        return Immediate.parse(self, string, hints)


class REUSE(ChoiceOf):
    def __init__(self, src, mode=None):
        name = f"[[{src}]]"
        if isinstance(mode, dict):
            name = f"[[dict({src})]]"
        elif isinstance(mode, str):
            name = f"[[{src}:{mode}]]"
        super().__init__(name)
        self.src = src
        self.mode = mode

    def parse(self, src, hints={}):
        if self.mode is None:
            return src
        if isinstance(self.mode, dict):
            choices = ()
            for v in src.choices:
                choices += self.mode.get(v, ())
            return src.dup(name=self.name, choices=choices, hints=hints)
        choices = getattr(src, 'choices', src)
        if self.mode == 'n':
            choices = getattr(src - 1, 'choices') + getattr(src + 1, 'choices')
        name = f"{{{','.join(map(str,choices))}}}"
        result = src.dup(name=name, choices=choices, hints=hints)
        return result


def repack(kwargs):
    kwargs = dict(kwargs)
    def repack_(*, rd=None, rs1=None, rsd=None, rs2=None, imm=None, rs_imm=None, **kwargs):
        if rs_imm: assert not (rs2 or imm)
        if rsd: assert not (rd or rs1)
        assert not (rs2 and imm)

        d = {
            'rsd': rsd,
            'rd': rd or rsd,
            'rs1': rs1 or rsd,
            'rs2': rs2,
            'imm': imm,
            'rs_imm': rs_imm or rs2 or imm,
        }
        kwargs.update({ k:v for k, v in d.items() if v })
        return kwargs
    return repack_(**kwargs)

def prettydict(kwargs):
    return { k: v.name for k,v in kwargs.items() }

class Instruction:
    def __init__(self, opcode, hints={}, **kwargs):
        if isinstance(opcode, str): opcode = Opcode(opcode)
        assert isinstance(opcode, (Opcode, REUSE)), f"op={opcode}, {repr(opcode)}"
        self.name = f"Instruction('{opcode.name}')"
        self.opcode = opcode
        bits = opcode.bits
        count = opcode.count
        arglist = []
        for k, v in kwargs.items():
            arglist.append(k)
            setattr(self, k, v)
            bits += v.bits
            count *= v.count
        self.bits = bits
        self.count = count

        if arglist:
            comma = '},{'
            auto_fmt = f"{{opcode:<7}} {{{comma.join(arglist)}}}"
        else:
            auto_fmt = f"{self.opcode.name}"

        if not hasattr(self, 'fmt'):
            self.fmt = formats.get(self.opcode.name, auto_fmt)

    def __str__(self):
        fmt_args = repack(vars(self))
        try:
            return self.fmt.format(**fmt_args)
        except KeyError as e:
            #raise KeyError(f"format {self.fmt=} - with {set(fmt_args.keys())}, exception={e}")
            raise KeyError(f"format {self.fmt=} - with {fmt_args}, exception={e}")

    def __format__(self, spec):
        return self.__str__().__format__(spec)

    def parse(self, opcode, operands, hints={}):
        opc = self.opcode.parse(opcode)
        operands = list(operands)
        # TODO: maybe look up values for hints here rather than externally?
        args = {}
        step = "wut?"
        try:
            if hasattr(self, 'rsd'):
                step = 'rsd'
                rd = operands.pop(0)
                rs1 = operands.pop(0)
                args['rsd'] = self.rsd.parse(rd, hints=hints)
                # Raise an error if rs1 doesn't match rd.
                args['rsd'].parse(rs1, hints=hints)
            else:
                step = 'not rsd'
                if hasattr(self, 'rd'):
                    step = 'rd'
                    args['rd'] = self.rd.parse(operands.pop(0), hints=hints)
                if hasattr(self, 'rs1'):
                    step = 'rs1'
                    args['rs1'] = self.rs1.parse(operands.pop(0), hints=hints)
            if hasattr(self, 'rs_imm'):
                step = 'rs_imm'
                tmp = self.rs_imm.parse(operands.pop(0), hints=hints)
                if isinstance(tmp, Register):
                    args['rs2'] = tmp
                else:
                    args['imm'] = tmp
            else:
                if hasattr(self, 'rs2'):
                    step = 'rs2'
                    args['rs2'] = self.rs2.parse(operands.pop(0), hints=hints)
                if hasattr(self, 'imm'):
                    step = 'imm'
                    args['imm'] = self.imm.parse(operands.pop(0), hints=hints)
            if operands:
                raise ValueError(f"Too many operands: {operands}")
        except Exception as e:
            raise type(e)(f"parsing {step=} with {hints=}: {e}")
        return Instruction(opc, **args, hints=hints)

    def specialise(self, ref, hints={}):
        # print(f"Forwarding '{ref}' arguments to '{self}', {hints=}")
        result = copy.copy(self)
        for k in vars(result):
            v = getattr(result, k)
            if isinstance(v, REUSE):
                alt = 'rsd' if v.src == 'rd' else 'nevermind'
                replacement = getattr(ref, v.src, getattr(ref, alt, None))
                v = v.parse(replacement, hints=hints)
                setattr(result, k, v)
                result.name += "/" + k
            else:
                if hasattr(v, 'shift'):
                    v.shift = hints.get('shift', v.shift)

        # print(f"Result of forwarding: {result}")
        return result

    def search(self, *args, **kwargs):
        if not (match := self.re_prog.search(*args, **kwargs)):
            return None
        for k, v in match.groupdict().items():
            validate = getattr(getattr(self, k, None), 'validate', None)
            if validate and not validate(v):
                return None
        return match


class ImplicitInstruction(Instruction):
    def __init__(self, opcode, **kwargs):
        super().__init__(opcode, **kwargs)

class Reserved(Instruction):
    opcode = Opcode("--reserved--")
    def __init__(self, bits, message=None):
        super().__init__(self.opcode)
        self.bits = bits
        self.count = 1 << bits
        self.fmt = message or "--reserved--"



class ari2(Instruction):
    opcode = Opcode("arith2", choices=(
        "addi",     # imm-2
        "add",
        "sub",
        "and",
    ))
    def __init__(self, **kwargs):
        super().__init__(self.opcode, **kwargs)

class ari3(Instruction):
    opcode = Opcode("arith3", choices=(
        "addi",     # imm+0
        "addi",     # imm+32
        "addi",     # imm-64
        "addi",     # imm-32
        "add",
        "sub",
        "and",
        "or",
    ))
    def __init__(self, **kwargs):
        super().__init__(self.opcode, **kwargs)

class ari4(Instruction):
    opcode = Opcode("arith4", choices=(
        "addi",     # imm+0
        "addi",     # imm-32
        "addiw",    # imm+0
        "addiw",    # imm-32
        "addi4spn", # imm+0
        "addi4spn", # imm+32
        "andi",     # imm+0
        "andi",     # imm-32
        "add",
        "addw",
        "sub",
        "subw",
        "and",
        "bic",
        "or",
        "xor",
    ))
    def __init__(self, **kwargs):
        super().__init__(self.opcode, **kwargs)

class ari5i(Instruction):
    opcode = Opcode("arith5i", choices=(
        "addi",     # imm+0
        "addi",     # imm-32
        "addiw",    # imm+0
        "addiw",    # imm-32
        "andi",     # imm+0
        "andi",     # imm-32
        "addi4spn", # imm+0
        "addi4spn", # imm+32
        "slli",     # imm+0
        "slli",     # imm+32
        "srli",     # imm+0
        "srli",     # imm+32
        "srai",     # imm+0
        "srai",     # imm+32
        "rsbi",     # imm+0
        "rsbi",     # imm+32
    ))
    def __init__(self, **kwargs):
        super().__init__(self.opcode, **kwargs)

class ari5r(Instruction):
    opcode = Opcode("arith5r", choices=(
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
# TODO: find space for "mv"?  It'll be partially redundant with some other shared operand cases -- could be messy.
    ))
    def __init__(self, **kwargs):
        super().__init__(self.opcode, **kwargs)

class ari5(Instruction):
    opcode = ari5i.opcode + ari5r.opcode
    opcode.name = "arith5"
    def __init__(self, **kwargs):
        super().__init__(self.opcode, **kwargs)

class cmp(Instruction):
    opcode = Opcode("cmpi", choices=(
        "slti",     # imm+0
        "slti",     # imm-32
        "sltiu",    # imm+0
        "sltiu",    # imm+32
        "seqi",     # imm+0
        "seqi",     # imm-32
        "bittesti", # imm+0
        "bittesti", # imm+32
    ))
    def __init__(self, **kwargs):
        super().__init__(self.opcode, **kwargs)

class ldst(Instruction):
    opcode = Opcode("ldst", choices=(
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
    ))
    fmt = LDST_format
    re = LDST_regex
    def __init__(self, **kwargs):
        super().__init__(self.opcode, **kwargs)


class LDST(ImplicitInstruction):
    fmt = ldst.fmt
    re = ldst.re
    opcode = REUSE("opcode")
    def __init__(self, **kwargs):
        super().__init__(self.opcode, **kwargs)


class pair(Instruction):
    opcode_pairs = [
        ("scarry", "add"),
        ("sub", "add"),
        ("min", "max"),
        ("minu", "maxu"),
        ("and", "bic"),
        ("mulhsu", "mul"),
        ("mulh", "mul"),
        ("mulhu", "mul"),
        ("div", "rem"),
        ("divu", "remu"),
        ("undef_a", "undef_b"),
        ("undef_c", "undef_d"),
    ]
    opcode = Opcode("pair.a",
        choices=sum(opcode_pairs, ()), roundoff=False)
    def __init__(self, **kwargs):
        super().__init__(self.opcode, **kwargs)


class PAIR(ImplicitInstruction):
    pair_dict = { k: (v,) for k,v in pair.opcode_pairs } | {
        "add":  ("sltu", "sub",),
        "max":  ("min",),
        "maxu": ("minu",),
        "bic":  ("and",),
        "mul":  ("mulh", "mulhu", "mulhsu",),
        "rem":  ("div",),
        "remu": ("divu",),
        "undef_b": ("undef_a",),
        "undef_d": ("undef_c",),
    }
    opcode = REUSE("opcode", pair_dict)
    def __init__(self, **kwargs):
        super().__init__(self.opcode, **kwargs)


ZERO = SpecificRegister(RegName.zero)
RA = SpecificRegister(RegName.ra)
SP = SpecificRegister(RegName.sp)
T6 = SpecificRegister(RegName.t6)


rd = Register("rd")
rd_nz = Register("rdnz", nonzero=True)
rsd = Register("rsd")
rsd_nz = Register("rsdnz", nonzero=True)
rs1 = Register("rs1")
rs2 = Register("rs2")

rd4 = Register4("rd")
rd4_nz = Register4("rdnz", nonzero=True)
rsd4 = Register4("rsd")
rsd4_nz = Register4("rsdnz", nonzero=True)
rs1_4 = Register4("rs1")
rs2_4 = Register4("rs2")
rs4_imm = RegImm("rs_imm", "rs2", "imm", choices=rsd4.choices)

rs_imm = RegImm("rs_imm", "rs2", "imm")
rd_3 = Register3("rd")
rsd_3 = Register3("rsd")
rs1_3 = Register3("rs1")
rs2_3 = Register3("rs2")

imm0 = SpecificImmediate(name="0", value=0)
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
imm24 = Immediate(24)

class InstructionSet:
    def __init__(self, name, instructions):
        self.name = name
        self.instructions = instructions
        self.hitrate = [0] * len(instructions)


def dump(inset : InstructionSet):
    size = 0
    for ops,hits in zip(inset.instructions, inset.hitrate):
        bits = 0
        count = 1
        display = []
        for op in ops:
            bits += op.bits
            count *= op.count
            display.append(f"{op.bits:2}: {str(op):<30}")

        print(f"{size:#11x}{count:+#11x}: {"  ".join(display)}  ({bits:2} bits)  {hits} hits")
        size += count

    print(f"total size: {size:#x},  bits: {(size - 1).bit_length()}")
    print()


def light_parse(line):
    for pat,repl in unaliases.items():
        line = re.sub(pat, repl, line)
    operands = filter(None, INSN_regex.match(line).groups())
    opcode = next(operands)
    operands = list(operands)
    return opcode,operands


INSN_regex = re.compile(r'.*\b(\w+)\s+(\w+)(?:,(?=-?\d+[(](\w+)[)])?(-?\w+))?(?:,(-?\w+))?(?:,(-?\w+))?(?:,(-?\w+))?', re.ASCII)
def first_line(inset : InstructionSet, line, verbose=False):
    opcode,operands = light_parse(line)
    result = []
    why = set()

    def printonce(*args, **kwargs):
        print(*args, **kwargs)
        while True: yield " "
    header = printonce("input:", line)

    no_matches = True
    for i in range(len(inset.instructions)):
        first = inset.instructions[i][0]
        stage = f"row {i}, {first}"
        second = None
        try:
            insn = first.parse(opcode, list(operands))
            second = inset.instructions[i][1]
            stage += " ; " + str(second)
            no_matches = False
            try:
                opcode_count = first.opcode.choices.count(opcode)
            except AttributeError:
                opcode_count = 1
            hints = {
                'opcode': opcode,
                'shift': wordsizes.get(opcode, 0),
                'signed': opcode not in unsigned_immediates,
                'extrabits': (opcode_count - 1).bit_length(),
            }
            stage += f" {hints=}"
            if (insn2 := second.specialise(insn, hints=hints)):
                if verbose: print(f"{stage}, prepared: {insn2}")
                result.append((insn2,i))
            else:
                if verbose: print(f"second instruction rejected {match.groupdict()}")
        except ValueError as e:
            if 'range' in str(e): why.add('range')
            why.add('value')
            if verbose: print(next(header), f"{stage}, value: {e}")
        except KeyError as e:
            why.add('key')
            if verbose: print(next(header), f"{stage}, key: {e}")

    if no_matches:
        why.add('nofirstmatch')
    return result, why


def next_line(patterns, line, verbose=False):
    opcode,operands = light_parse(line)

    for template,i in patterns:
        try:
            if (insn := template.parse(opcode, operands)):
                return insn,i
        except ValueError as e:
            if verbose: print(f"No match on {template} for: {opcode=}, {operands=}, val: {e}")
            pass
        except KeyError as e:
            if verbose: print(f"No match on {template} for: {opcode=}, {operands=}, key: {e}")
            pass
    return None


def try_pair(inset : InstructionSet, line0, line1):
    patterns,why = first_line(inset, line0, verbose=True)
    if why: print("  input1 problems:", why)
    for pat in patterns:
        print(f"  pattern, row={pat[1]:2}: {pat[0]}")

    print("input0:", line0)
    print("input1:", line1)
    pat = next_line(patterns, line1, verbose=True)
    print("success" if pat else "failure")
    print()


def compress(inset : InstructionSet, filename, verbose=True, quiet=False):
    saved = 0
    total = 0
    with open(filename, "rt") as f:
        prev = None
        patterns = []
        for line in f:
            line = line.strip()
            if not line.startswith('0x'):
                if verbose and not quiet: print('#', line)
                continue

            if (pat := next_line(patterns, line)):
                if verbose and not quiet:
                    print("..", prev)
                    print("^^", line)
                prev = None
                patterns = []
                saved += 1
                inset.hitrate[pat[1]] += 1
            else:
                if prev and not quiet: print(f"{len(patterns):2} {prev}")
                prev = None
                failed_patterns = patterns
                patterns, why = first_line(inset, line)
                if patterns:
                    prev = line
                else:
                    if not quiet: print("XX", line, ' - ', why)
            total += 1

    dump(inset)
    print(f"{saved=}, {total=}")


rvc = InstructionSet("rvc", [
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
])

my_attempt = InstructionSet("my_attempt", [
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

    ( ldst(rd=rd, rs1=rs1, imm=imm10),          LDST(rd=REUSE("rd", 'n'), rs1=REUSE("rs1"), imm=REUSE("imm", 'n')), ),

    ( ari3(rsd=rsd, rs_imm=rs_imm),             ldst(rd=rd, rs1=rs1, imm=imm0), ),
    ( ldst(rd=rd, rs1=rs1, imm=imm0),           ari3(rsd=rsd, rs_imm=rs_imm), ),
])

attempt2 = InstructionSet("my second attempt", [
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

    ( ldst(rd=rd, rs1=rs1, imm=imm5),           LDST(rd=rd, rs1=REUSE("rs1"), imm=REUSE("imm", 'n')), ),

    ( ari3(rsd=rsd, rs_imm=rs_imm),             ldst(rd=rd, rs1=rs1, imm=imm0), ),
    ( ldst(rd=rd, rs1=rs1, imm=imm0),           ari3(rsd=rsd, rs_imm=rs_imm), ),
])

attempt3 = InstructionSet("my third attempt", [
    ( ari2(rd=rd4, rs1=rs1_4, rs_imm=rs4_imm),  ari2(rd=rd4, rs1=rs1_4, rs_imm=rs4_imm), ),
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

    ( ari5(rsd=rsd4, rs_imm=rs_imm),            Instruction("sw", rd=REUSE("rd"), rs1=SP, imm=imm8), ),
    ( ari5(rsd=rsd4, rs_imm=rs_imm),            Instruction("sd", rd=REUSE("rd"), rs1=SP, imm=imm8), ),
    ( Instruction("lw", rd=rd4, rs1=SP, imm=imm8), ari5(rsd=REUSE("rd"), rs_imm=rs_imm), ),
    ( Instruction("ld", rd=rd4, rs1=SP, imm=imm8), ari5(rsd=REUSE("rd"), rs_imm=rs_imm), ),

    ( Reserved(21), ),

    ( pair(rd=rd, rs1=rs1, rs2=rs2),            PAIR(rd=rd, rs1=REUSE("rs1"), rs2=REUSE("rs2")), ),

    ( ldst(rd=rd, rs1=rs1, imm=imm5),           LDST(rd=rd, rs1=REUSE("rs1"), imm=REUSE("imm", 'n')), ),

    ( ari3(rsd=rsd, rs_imm=rs_imm),             ldst(rd=rd, rs1=rs1, imm=imm0), ),
    ( ldst(rd=rd, rs1=rs1, imm=imm0),           ari3(rsd=rsd, rs_imm=rs_imm), ),
])

## Do some stuff

print('---------\n\n')
dump(rvc)
dump(my_attempt)
dump(attempt2)
dump(attempt3)
print('---------\n\n')

test_pairs = (
    ("slli  a4,a1,48"       ,"srli  a4,a4,48"),
    ("add   a0,a1,a2"       ,"add   a3,a4,a5"),
    ("ld    a0,136(s1)"     ,"lw    a4,-1888(tp)"),
    ("mv    a0,a1"          ,"ret"),
    ("mv    a2,123"         ,"ret"),
    ("ld    ra,152(sp)"     ,"ld    s0,144(sp)"),
    ("xor   a5,a5,a4"       ,"bnez  a5,242"),

    ("mv    a0,s10"         ,"addi  sp,sp,-16"),
    ("sd    s0,8(sp)"       ,"addi  s0,sp,16"),

    ("max   a3,a5,a4"       ,"min   a2,a5,a4"),
    ("ld    a0,0(s1)"       ,"add   s1,s1,31"),
    ("ld    a0,0(s1)"       ,"add   s1,s1,32"),
    ("ld    a0,0(s1)"       ,"add   s1,s1,1024"),
    ("lb    a0,0(s1)"       ,"add   s1,s1,31"),
    ("lb    a0,0(s1)"       ,"add   s1,s1,32"),
    ("ld    a0,0(s1)"       ,"add   s1,s1,1024"),
)

if True:
    for left, right in test_pairs:
        try_pair(attempt2, left, right)
    print('---------\n\n')

if True:
    try:
        compress(attempt2, "qemu-lite.txt")
        compress(attempt3, "qemu-lite.txt", quiet=True)
    except KeyboardInterrupt:
        print("stopped by ^C")

