import math
import re

DEFAULT_IMM_SHIFT = 3

LDST_regex = r"\b{opcode}\s+{rd},\s*{imm}[(]{rs1}[)]"
LDSTSP_regex = r"\b{opcode}\s+{rd},\s*{imm}[(]sp[)]"
REG_regex = r'zero|[astx][0123]?\d|[sgtf]p|ra'
IMM_regex = r'[-+]?\d+'

unaliases = {
    fr"\bmv\s+(?P<rd>{REG_regex}),\s*(?P<rs2>{REG_regex})":   r"add     \g<rd>,zero,\g<rs2>",
    fr"\bmv\s+(?P<rd>{REG_regex}),\s*(?P<imm>{IMM_regex})":   r"addi    \g<rd>,zero,\g<imm>",
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

class Opcode:
    bits = 0
    count = 1

    name = "generic name"
    def __init__(self, name, opcodes=(), roundoff=True, aliases=()):
        self.name = name
        self.choices = opcodes or (name,)
        self.count = len(self.choices)
        self.aliases = set(self.choices) | set(aliases)
        self.bits = (self.count - 1).bit_length()
        if roundoff:
            self.count = 1 << self.bits
        self.re = f"{'|'.join(self.aliases)}"

    def __str__(self):
        return self.name
    def __format__(self, spec):
        if spec == 'pair':
            try:
                return pair.opcode_dict[self.name]
            except KeyError as e:
                raise KeyError(f"No '{self.name}' in {set(pair.opcode_dict.keys())}: {e}")
        return self.__str__().__format__(spec)
    def __add__(self, other):
        return Opcode(
                name="+".join((self.name, other.name)),
                opcodes=tuple(tuple(self.choices) + tuple(other.choices)),
                aliases=(self.aliases | other.aliases)
        )

    def parse(self, string, hints={}):
        if string not in self.aliases:
            raise ValueError(f"Opcode {string} not in {self.aliases}")
        return Opcode(string)


def repack(kwargs):
    kwargs = dict(kwargs)
    def repack_(*, rd=None, rs1=None, rsd=None, rs2=None, imm=None, rs_imm=None, **kwargs):
        if rs_imm: assert not (rs2 or imm)
        if rsd: assert not (rd or rs1)
        assert not (rs2 and imm)

        d = {
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
    def __init__(self, opcode, **kwargs):
        if isinstance(opcode, str): opcode = Opcode(opcode)
        assert isinstance(opcode, Opcode), f"op={opcode}, {repr(opcode)}"
        self.name = f"Instruction('{opcode.name}')"
        self.opcode = opcode
        bits = opcode.bits
        count = opcode.count
        arglist = []
        unique = set()
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
            comma = '},{'
            auto_fmt = f"{{opcode:<7}} {{{comma.join(arglist)}}}"
            re_comma = r'},\s*{'
            auto_re = fr"\b{{opcode}}\s+{{{re_comma.join(arglist)}}}"
        else:
            auto_fmt = f"{self.opcode.name}"
            auto_re = f"{self.opcode.name}"

        if not hasattr(self, 'fmt'):
            self.fmt = formats.get(self.opcode.name, auto_fmt)

        if not hasattr(self, 're'):
            self.re = regexps.get(self.opcode.name, auto_re)

        def capture_fmt(fmt, **kwargs):
            expanded = { k: f"(?P<{k}>{v.re})" for k,v in kwargs.items() }
            return fmt.format(**expanded)

        def no_capture_fmt(fmt, **kwargs):
            expanded = { k: f"({v.re})" for k,v in kwargs.items() }
            return fmt.format(**expanded)


        self.re_nocapture = no_capture_fmt(self.re, opcode=self.opcode, **repack(kwargs))
        self.re = capture_fmt(self.re, opcode=self.opcode, **repack(kwargs))
        self.re_prog = re.compile(self.re, re.ASCII)

    def __str__(self):
        fmt_args = repack(vars(self))
        try:
            return self.fmt.format(**fmt_args)
        except KeyError as e:
            #raise KeyError(f"format {self.fmt=} - with {set(fmt_args.keys())}, exception={e}")
            raise KeyError(f"format {self.fmt=} - with {fmt_args}, exception={e}")

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

    def re_cooked(self, fmt_args):
        fmt_args = repack(fmt_args)
        fmt_re = self.re_nocapture
        try:
            return re.compile(fmt_re.format(**fmt_args), re.ASCII)
        except KeyError as e:
            raise KeyError(f"re_cooked {fmt_re=} - with {set(fmt_args.keys())}, exception={e}")


class Register:
    choices = (
        "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2",
        "s0", "s1", "a0", "a1", "a2", "a3", "a4", "a5",
        "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7",
        "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6",
    )
    aliases = set(choices) | {
        "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7",
        "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15",
        "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23",
        "x24", "x25", "x26", "x27", "x28", "x29", "x30", "x31",
        "fp",
    }
    #re = '|'.join(aliases)
    re = REG_regex
    def __init__(self, name, count = 32):
        self.name = name
        self.count = count
        self.bits = (self.count - 1).bit_length()
        if count == 1:
            self.choices = { name }
            self.aliases = self.choices  # TODO: look up correct aliases
            self.re = name

    def __str__(self):
        return self.name
    def __format__(self, spec):
        if spec == 'pair':
            if self.name == 'fp': return r"s1"
            i = Register.choices.index(self.name)
            return Register.choices[i^1]
        if spec == 'next':
            if self.name == 'fp': return r"t2|s1"
            i = Register.choices.index(self.name)
            if not (0 < i and i < len(Register.choices) - 1):
                raise ValueError(f"Can't offer next register for register {i}")
            return "|".join((Register.choices[i-1], Register.choices[i+1]))
        return self.__str__().__format__(spec)

    def parse(self, string, hints={}):
        if string not in self.aliases:
            raise ValueError(f"Register {string} not in {self.aliases}")
        return Register(string, 1)


class Register3(Register):
    choices = ( "s0", "s1", "a0", "a1", "a2", "a3", "a4", "a5", )
    aliases = set(choices) | { "x8", "fp", "x9", "x10", "x11", "x12", "x13", "x14", "x15", }
    re = '|'.join(aliases)
    def __init__(self, name):
        super().__init__(name, 8)

class Immediate:
    re = IMM_regex
    shift = 0
    signed = True

    def __init__(self, size, name=None, hints={}):
        self.name = name or f"imm{size}"
        self.bits = size
        self.count = 1 << size
        self.shift = hints.get('shift', DEFAULT_IMM_SHIFT)
        self.signed = hints.get('signed', True)

    def __str__(self):
        return self.name
    def __format__(self, spec):
        if spec == 'next':
            value = int(self.name)
            k = 1 << self.shift
            return (f"{value-k}|{value+k}")
        return self.__str__().__format__(spec)

    def parse(self, string, hints={}):
        value = int(string)

        k = 1 << hints.get('shift', self.shift)
        signed = hints.get('signed', self.signed)
        bits = (self.bits + hints.get('extrabits', 0))

        if self.bits > 0:  # No need to check range if we're working with implicit immediate
            range_ = k << bits
            if signed:
                range_ //= 2
                if not (-range_ <= value and value < range_):
                    raise ValueError(f"Immediate {string} is out of range for {bits}-bit signed immediate {hints=}")
            else:
                if not (0 <= value and value < range_):
                    raise ValueError(f"Immediate {string} is out of range for {bits}-bit unsigned immediate {hints=}")
            # if value % k != 0:
            #     raise ValueError(f"Immediate {string} is not a multiple of {k}")

        retval = Immediate(size=0, name=string, hints=hints)
        retval.re = str(value)
        return retval


class RegImm(Register, Immediate):
    re = f'{Register.re}|{Immediate.re}'
    def __init__(self, name, reg_name, imm_name='imm', count=32, hints={}):
        super().__init__(name, count)
        self.shift = hints.get('shift', self.shift)
        self.signed = hints.get('signed', True)
        re = fr'((?P<{reg_name}>{Register.re})|(?P<{imm_name}>{Immediate.re}))'

    def parse(self, string, hints={}):
        try:
            return Register.parse(self, string, hints)
        except ValueError:
            pass
        return Immediate.parse(self, string, hints)

class ari3(Instruction):
    opcode = Opcode("arith3", (
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
    opcode = Opcode("arith4", (
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
    opcode = Opcode("arith5i", (
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
    opcode = Opcode("arith5r", (
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
    opcode = Opcode("cmpi", (
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
    opcode = Opcode("ldst", (
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
    opcode_dict = { k: v for k,v in opcode_pairs } | {
        "add": "sltu|sub",
        "max": "min",
        "maxu": "minu",
        "bic": "and",
        "mul": "mulhs?u?",
        "rem": "div",
        "remu": "divu",
        "undef_b": "undef_a",
        "undef_d": "undef_c",
    }
    opcode = Opcode("pair.a",
        opcodes=tuple(p[0] for p in opcode_pairs),
        aliases=set(opcode_dict.keys()),
        roundoff=False)
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
    opcode = Opcode("{opcode}")
    fmt = ldst.fmt
    re = ldst.re
    def __init__(self, **kwargs):
        super().__init__(self.opcode, **kwargs)


class PAIR(ImplicitInstruction):
    opcode = Opcode("{opcode:pair}")
    def __init__(self, **kwargs):
        super().__init__(self.opcode, **kwargs)


ZERO = ImplicitRegister("zero")
RA = ImplicitRegister("ra")
SP = ImplicitRegister("sp")
T6 = ImplicitRegister("t6")


rd = Register("rd")
rd_nz = Register("rd", 31)
rsd = Register("rsd")
rsd_nz = Register("rsd", 31)
rs1 = Register("rs1")
rs2 = Register("rs2")

rs_imm = RegImm("rs_imm", "rs2", "imm")
rd_3 = Register3("rd")
rsd_3 = Register3("rsd")
rs1_3 = Register3("rs1")
rs2_3 = Register3("rs2")

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

        print(f"{size:#10x}{count:+#11x}: {"  ".join(display)}  ({bits:2} bits)  {hits} hits")
        size += count

    print(f"total size: {size:#x},  bits: {(size - 1).bit_length()}")
    print()


def first_line(inset : InstructionSet, line, verbose=False):
    result = []
    why = set()
    for pat,repl in unaliases.items():
        line = re.sub(pat, repl, line)

    def printonce(*args, **kwargs):
        print(*args, **kwargs)
        while True: yield " "
    header = printonce("input:", line)

    no_matches = True
    for i in range(len(inset.instructions)):
        first = inset.instructions[i][0]
        if (match := first.search(line)):
            no_matches = False
            opcode = match['opcode']
            second = inset.instructions[i][1]
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
            if verbose: print(f"{hints=}")
            try:
                args = {}
                for k, v in match.groupdict().items():
                    if hasattr(first, k):
                        args[k] = getattr(first, k).parse(v, hints=hints)
                if verbose: print(f"Match: {first.name}, {prettydict(args)=}")
                if (regex := second.re_cooked(args)):
                    result.append((regex,i))
                else:
                    if verbose: print(f"second instruction rejected {match.groupdict()}")
            except ValueError as e:
                if 'range' in str(e): why.add('range')
                why.add('value')
                if verbose: print(next(header), f"parse failure on match {first.name}: {e}")
            except KeyError as e:
                why.add('construct')
                print(next(header), f"construction error for {second.name}: {e}")
    if no_matches:
        why.add('nomatch')
    return result, why


def next_line(patterns, line):
    for pattern,replace in unaliases.items():
        line = re.sub(pattern, replace, line)
    for pat in patterns:
        if pat[0].search(line):
            return pat
    return None


def try_pair(inset : InstructionSet, line0, line1):
    patterns,why = first_line(inset, line0, verbose=True)
    if why: print("  input1 problems:", why)
    for pat in patterns:
        print('  pattern:', pat[0].pattern)

    print("input0:", line0)
    print("input1:", line1)
    pat = next_line(patterns, line1)
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

    ( ldst(rd=rd, rs1=rs1, imm=imm10),          LDST(rd=REUSE("rd:next"), rs1=REUSE("rs1"), imm=REUSE("imm:next")), ),

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

    ( ldst(rd=rd, rs1=rs1, imm=imm5),           LDST(rd=rd, rs1=REUSE("rs1"), imm=REUSE("imm:next")), ),

    ( ari3(rsd=rsd, rs_imm=rs_imm),             ldst(rd=rd, rs1=rs1, imm=imm0), ),
    ( ldst(rd=rd, rs1=rs1, imm=imm0),           ari3(rsd=rsd, rs_imm=rs_imm), ),
])

## Do some stuff

dump(rvc)
dump(my_attempt)
dump(attempt2)
try_pair(attempt2, " slli a4,a1,48", " srli a4,a4,48")
try_pair(attempt2, " ld a0,136(s1)", " lw a4,-1888(tp)")
try_pair(attempt2, " mv a0,a1", " ret")
try_pair(attempt2, " mv a2,123", " ret")
try_pair(attempt2, " ld ra,152(sp)", " ld s0,144(sp)")
try_pair(attempt2, "xor a5,a5,a4", "bnez a5,242")

try_pair(attempt2, "mv   a0,s10", "`addi sp,sp,-16")
try_pair(attempt2, "sd   s0,8(sp)", "`addi s0,sp,16")

try_pair(attempt2, "max a3,a5,a4", "min a2,a5,a4")
print('---------\n\n')

try:
    compress(attempt2, "qemu-lite.txt")
    compress(my_attempt, "qemu-lite.txt", quiet=True)
except KeyboardInterrupt:
    print("stopped by ^C")

