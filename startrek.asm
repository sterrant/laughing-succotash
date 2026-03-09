; -----------------------------------------------------------------------------
; STARTREK.ASM  -  Tiny 8080 "Super Star Trek" style game for IMSAI 8080
; -----------------------------------------------------------------------------
; Assembler target : CP/M ASM
; Load address     : 0000h
; Console hardware : Compupro Interfacer 1 (Port A data=00h, status/control=01h)
;
; Serial assumptions:
;   Status bit 0 = RX data ready
;   Status bit 1 = TX data register empty
;
; Commands:
;   S = status report
;   W = warp (N/S/E/W then distance 1..3)
;   P = fire phasers
;   T = fire photon torpedo
;   H = help
;   Q = quit
; -----------------------------------------------------------------------------

DATA    EQU     00H
STAT    EQU     01H

        ORG     0000H

START:  LXI     SP,0F000H
        MVI     A,05AH
        STA     SEED

        MVI     A,03H
        STA     DAYSLEFT

        MVI     A,08H
        STA     XPOS
        MVI     A,04H
        STA     YPOS

        CALL    RND18
        STA     KX
        CALL    RND18
        STA     KY

        CALL    RND18
        STA     BX
        CALL    RND18
        STA     BY

        MVI     A,58H         ; 600 decimal = 0258h
        STA     KHP
        MVI     A,02H
        STA     KHP+1

        MVI     A,0B8H        ; 3000 decimal = 0BB8h
        STA     ENERGY
        MVI     A,0BH
        STA     ENERGY+1

        MVI     A,00H
        STA     SHIELDS
        STA     SHIELDS+1

        MVI     A,0AH
        STA     TORPS

        LXI     H,MSG_TITLE
        CALL    PRINT
        LXI     H,MSG_INTRO
        CALL    PRINT

MAINLP: CALL    CHECK_END
        LXI     H,MSG_CMD
        CALL    PRINT
        CALL    GETCH_UP
        MOV     B,A
        CALL    PUTCH
        CALL    CRLF

        MOV     A,B
        CPI     'S'
        JZ      DO_STATUS
        CPI     'W'
        JZ      DO_WARP
        CPI     'P'
        JZ      DO_PHASER
        CPI     'T'
        JZ      DO_TORP
        CPI     'H'
        JZ      DO_HELP
        CPI     'Q'
        JZ      DO_QUIT

        LXI     H,MSG_WHAT
        CALL    PRINT
        JMP     MAINLP

DO_HELP:LXI     H,MSG_HELP
        CALL    PRINT
        JMP     TURN_DONE

DO_STATUS:
        CALL    STATUS
        JMP     TURN_DONE

DO_WARP:
        CALL    WARP
        JMP     TURN_DONE

DO_PHASER:
        CALL    PHASER
        JMP     TURN_DONE

DO_TORP:
        CALL    TORPEDO
        JMP     TURN_DONE

DO_QUIT:
        LXI     H,MSG_BYE
        CALL    PRINT
        HLT

TURN_DONE:
        CALL    ENEMY_TURN
        CALL    DEC_DAY
        JMP     MAINLP

; -----------------------------------------------------------------------------
; Status and game flow
; -----------------------------------------------------------------------------
STATUS: LXI     H,MSG_STATUS
        CALL    PRINT

        LXI     H,MSG_POS
        CALL    PRINT
        LDA     XPOS
        CALL    PRTDIG
        MVI     A,','
        CALL    PUTCH
        LDA     YPOS
        CALL    PRTDIG
        CALL    CRLF

        LXI     H,MSG_ENE
        CALL    PRINT
        LHLD    ENERGY
        CALL    PRT16
        CALL    CRLF

        LXI     H,MSG_SHD
        CALL    PRINT
        LHLD    SHIELDS
        CALL    PRT16
        CALL    CRLF

        LXI     H,MSG_TRP
        CALL    PRINT
        LDA     TORPS
        CALL    PRTDIG2
        CALL    CRLF

        LXI     H,MSG_KPOS
        CALL    PRINT
        LDA     KX
        CALL    PRTDIG
        MVI     A,','
        CALL    PUTCH
        LDA     KY
        CALL    PRTDIG
        CALL    CRLF

        LXI     H,MSG_KHP
        CALL    PRINT
        LHLD    KHP
        CALL    PRT16
        CALL    CRLF

        LXI     H,MSG_DAY
        CALL    PRINT
        LDA     DAYSLEFT
        CALL    PRTDIG2
        CALL    CRLF
        RET

CHECK_END:
        LHLD    KHP
        MOV     A,H
        ORA     L
        JNZ     CHK_DAY
        LXI     H,MSG_WIN
        CALL    PRINT
        HLT
CHK_DAY:LDA     DAYSLEFT
        ORA     A
        JNZ     CHK_EN
        LXI     H,MSG_LOSE_TIME
        CALL    PRINT
        HLT
CHK_EN: LHLD    ENERGY
        MOV     A,H
        ORA     L
        JNZ     CHK_OK
        LXI     H,MSG_LOSE_EN
        CALL    PRINT
        HLT
CHK_OK: RET

DEC_DAY: LDA     DAYSLEFT
        ORA     A
        RZ
        DCR     A
        STA     DAYSLEFT
        RET

; -----------------------------------------------------------------------------
; Warp command
; -----------------------------------------------------------------------------
WARP:   LXI     H,MSG_DIR
        CALL    PRINT
        CALL    GETCH_UP
        MOV     B,A
        CALL    PUTCH
        CALL    CRLF

        LXI     H,MSG_DIST
        CALL    PRINT
        CALL    GETCH_UP
        CALL    PUTCH
        CALL    CRLF
        SUI     '0'
        JC      W_BAD
        CPI     04H
        JNC     W_BAD
        MOV     C,A

        MOV     A,B
        CPI     'N'
        JZ      W_N
        CPI     'S'
        JZ      W_S
        CPI     'E'
        JZ      W_E
        CPI     'W'
        JZ      W_W
W_BAD:  LXI     H,MSG_BADCMD
        CALL    PRINT
        RET

W_N:    LDA     YPOS
        MOV     B,A
WNLP:   DCR     B
        JZ      W_BOUND
        DCR     C
        JNZ     WNLP
        MOV     A,B
        STA     YPOS
        JMP     W_DONE

W_S:    LDA     YPOS
        MOV     B,A
WSLP:   INR     B
        MOV     A,B
        CPI     09H
        JNC     W_BOUND
        DCR     C
        JNZ     WSLP
        MOV     A,B
        STA     YPOS
        JMP     W_DONE

W_E:    LDA     XPOS
        MOV     B,A
WELP:   INR     B
        MOV     A,B
        CPI     09H
        JNC     W_BOUND
        DCR     C
        JNZ     WELP
        MOV     A,B
        STA     XPOS
        JMP     W_DONE

W_W:    LDA     XPOS
        MOV     B,A
WWLP:   DCR     B
        JZ      W_BOUND
        DCR     C
        JNZ     WWLP
        MOV     A,B
        STA     XPOS
        JMP     W_DONE

W_BOUND:
        LXI     H,MSG_BOUND
        CALL    PRINT
        RET

W_DONE: LXI     H,MSG_WARPED
        CALL    PRINT
        CALL    USE_ENERGY_50
        RET

; -----------------------------------------------------------------------------
; Phaser and torpedo
; -----------------------------------------------------------------------------
PHASER: LXI     H,MSG_PHASER
        CALL    PRINT
        CALL    DISTANCE
        MOV     A,B
        CPI     04H
        JC      PH_OK
        LXI     H,MSG_TOO_FAR
        CALL    PRINT
        RET
PH_OK:  CALL    USE_ENERGY_200
        JC      PH_NOE

        CALL    RAND8
        ANI     7FH
        ADI     64H            ; +100
        MOV     C,A

        LHLD    KHP
        MOV     A,L
        SUB     C
        MOV     L,A
        MOV     A,H
        SBB     00H
        MOV     H,A
        JNC     PH_SV
        LXI     H,0000H
PH_SV:  SHLD    KHP

        LXI     H,MSG_HIT
        CALL    PRINT
        MOV     A,C
        CALL    PRTDIG2
        CALL    CRLF
        RET
PH_NOE: LXI     H,MSG_NO_EN
        CALL    PRINT
        RET

TORPEDO:
        LDA     TORPS
        ORA     A
        JNZ     TP_OK
        LXI     H,MSG_NO_T
        CALL    PRINT
        RET
TP_OK:  DCR     A
        STA     TORPS

        LDA     XPOS
        MOV     B,A
        LDA     KX
        CMP     B
        JZ      TP_HIT
        LDA     YPOS
        MOV     B,A
        LDA     KY
        CMP     B
        JZ      TP_HIT

        LXI     H,MSG_MISS
        CALL    PRINT
        RET

TP_HIT: LXI     H,0000H
        SHLD    KHP
        LXI     H,MSG_T_HIT
        CALL    PRINT
        RET

; -----------------------------------------------------------------------------
; Enemy action
; -----------------------------------------------------------------------------
ENEMY_TURN:
        LHLD    KHP
        MOV     A,H
        ORA     L
        RZ

        CALL    DISTANCE
        MOV     A,B
        CPI     05H
        RNC

        CALL    RAND8
        ANI     3FH
        ADI     32H            ; 50..113
        MOV     C,A

        ; shields first
        LHLD    SHIELDS
        MOV     A,H
        ORA     L
        JZ      EA_ENERGY

        MOV     A,L
        SUB     C
        MOV     L,A
        MOV     A,H
        SBB     00H
        MOV     H,A
        JNC     EA_SSAVE
        LXI     H,0000H
EA_SSAVE:
        SHLD    SHIELDS
        JMP     EA_MSG

EA_ENERGY:
        LHLD    ENERGY
        MOV     A,L
        SUB     C
        MOV     L,A
        MOV     A,H
        SBB     00H
        MOV     H,A
        JNC     EA_ESAVE
        LXI     H,0000H
EA_ESAVE:
        SHLD    ENERGY

EA_MSG: LXI     H,MSG_UNDER
        CALL    PRINT
        MOV     A,C
        CALL    PRTDIG2
        CALL    CRLF
        RET

; -----------------------------------------------------------------------------
; Utilities
; -----------------------------------------------------------------------------
DISTANCE:
        LDA     XPOS
        MOV     B,A
        LDA     KX
        SUB     B
        JNC     D1
        CMA
        INR     A
D1:     MOV     B,A

        LDA     YPOS
        MOV     C,A
        LDA     KY
        SUB     C
        JNC     D2
        CMA
        INR     A
D2:     ADD     B
        MOV     B,A
        RET

USE_ENERGY_50:
        LHLD    ENERGY
        LXI     D,0032H
        CALL    SUBHLDE
        SHLD    ENERGY
        RET

USE_ENERGY_200:
        LHLD    ENERGY
        LXI     D,00C8H
        ; check HL >= DE
        MOV     A,H
        CMP     D
        JC      UEF
        JNZ     UEDO
        MOV     A,L
        CMP     E
        JC      UEF
UEDO:   CALL    SUBHLDE
        SHLD    ENERGY
        ORA     A               ; clear carry
        RET
UEF:    STC
        RET

SUBHLDE:
        MOV     A,L
        SUB     E
        MOV     L,A
        MOV     A,H
        SBB     D
        MOV     H,A
        RET

RAND8:  LDA     SEED
        MOV     B,A
        ADD     A               ; 2x
        ADD     A               ; 4x
        ADD     B               ; 5x
        INR     A               ; +1
        STA     SEED
        RET

RND18:  CALL    RAND8
        ANI     07H
        INR     A
        RET

GETCH_UP:
GC1:    IN      STAT
        ANI     01H
        JZ      GC1
        IN      DATA
        CPI     'a'
        RC
        CPI     07BH
        RNC
        SUI     20H
        RET

PUTCH:  MOV     C,A
PC1:    IN      STAT
        ANI     02H
        JZ      PC1
        MOV     A,C
        OUT     DATA
        RET

CRLF:   MVI     A,0DH
        CALL    PUTCH
        MVI     A,0AH
        CALL    PUTCH
        RET

PRINT:  MOV     A,M
        ORA     A
        RZ
        CALL    PUTCH
        INX     H
        JMP     PRINT

PRTDIG: ADI     '0'
        CALL    PUTCH
        RET

PRTDIG2:
        ; print 0..99 decimal from A
        MOV     B,A
        MVI     C,00H
PD2L:   MOV     A,B
        CPI     0AH
        JC      PD2O
        SUI     0AH
        MOV     B,A
        INR     C
        JMP     PD2L
PD2O:   MOV     A,C
        ADI     '0'
        CALL    PUTCH
        MOV     A,B
        ADI     '0'
        CALL    PUTCH
        RET

PRT16:  ; print HL as 4 decimal digits (0000..9999)
        MVI     B,00H
        LXI     D,03E8H        ; 1000
        CALL    DIGCT
        MOV     A,B
        ADI     '0'
        CALL    PUTCH

        MVI     B,00H
        LXI     D,0064H        ; 100
        CALL    DIGCT
        MOV     A,B
        ADI     '0'
        CALL    PUTCH

        MVI     B,00H
        LXI     D,000AH        ; 10
        CALL    DIGCT
        MOV     A,B
        ADI     '0'
        CALL    PUTCH

        MOV     A,L
        ADI     '0'
        CALL    PUTCH
        RET

DIGCT:  ; count how many DE fit in HL, result in B
DG1:    MOV     A,H
        CMP     D
        JC      DGX
        JNZ     DGS
        MOV     A,L
        CMP     E
        JC      DGX
DGS:    CALL    SUBHLDE
        INR     B
        JMP     DG1
DGX:    RET

; -----------------------------------------------------------------------------
; Data
; -----------------------------------------------------------------------------
MSG_TITLE:
        DB '*** IMSAI 8080 SUPER STARTREK (ML) ***',0DH,0AH,0
MSG_INTRO:
        DB 'Destroy the Klingon before stardate expires.',0DH,0AH,0
MSG_CMD:
        DB 0DH,0AH,'Command (S,W,P,T,H,Q)? ',0
MSG_WHAT:
        DB 'Unknown command.',0DH,0AH,0
MSG_HELP:
        DB 'S=status W=warp P=phaser T=torpedo H=help Q=quit',0DH,0AH,0
MSG_STATUS:
        DB '--- STATUS ---',0DH,0AH,0
MSG_POS:
        DB 'Enterprise sector : ',0
MSG_ENE:
        DB 'Energy            : ',0
MSG_SHD:
        DB 'Shields           : ',0
MSG_TRP:
        DB 'Torpedoes         : ',0
MSG_KPOS:
        DB 'Klingon sector    : ',0
MSG_KHP:
        DB 'Klingon strength  : ',0
MSG_DAY:
        DB 'Days left         : ',0
MSG_DIR:
        DB 'Direction (N/S/E/W)? ',0
MSG_DIST:
        DB 'Distance (1-3)? ',0
MSG_BADCMD:
        DB 'Bad warp parameters.',0DH,0AH,0
MSG_BOUND:
        DB 'Navigation boundary reached.',0DH,0AH,0
MSG_WARPED:
        DB 'Warp complete.',0DH,0AH,0
MSG_PHASER:
        DB 'Phasers fired (200 units).',0DH,0AH,0
MSG_TOO_FAR:
        DB 'Target out of effective phaser range.',0DH,0AH,0
MSG_HIT:
        DB 'Hit for ',0
MSG_NO_EN:
        DB 'Insufficient energy.',0DH,0AH,0
MSG_NO_T:
        DB 'No torpedoes remaining.',0DH,0AH,0
MSG_MISS:
        DB 'Torpedo missed.',0DH,0AH,0
MSG_T_HIT:
        DB 'Direct hit! Klingon destroyed.',0DH,0AH,0
MSG_UNDER:
        DB 'Enemy attack: damage ',0
MSG_WIN:
        DB 0DH,0AH,'Mission success. Federation saved.',0DH,0AH,0
MSG_LOSE_TIME:
        DB 0DH,0AH,'Mission failed. Stardate expired.',0DH,0AH,0
MSG_LOSE_EN:
        DB 0DH,0AH,'Mission failed. Energy depleted.',0DH,0AH,0
MSG_BYE:
        DB 0DH,0AH,'Leaving command.',0DH,0AH,0

SEED:   DB 00H
DAYSLEFT: DB 00H
XPOS:   DB 00H
YPOS:   DB 00H
KX:     DB 00H
KY:     DB 00H
BX:     DB 00H
BY:     DB 00H
TORPS:  DB 00H
ENERGY: DW 0000H
SHIELDS:DW 0000H
KHP:    DW 0000H

        END
