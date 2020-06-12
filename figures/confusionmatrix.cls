\newcommand{\MakeConfusion}[4]{
    \begin{tikzpicture}[
        box/.style={draw,rectangle,minimum size=2cm,text width=1.5cm,align=center}]
        
        \FPeval{\actualidle}{clip(#1+#2)}
        \FPeval{\actualspeech}{clip(#3+#4)}
        \FPeval{\predidle}{clip(#1+#3)}
        \FPeval{\predspeech}{clip(#2+#4)}
        
        \pgfmathparse{(#1/\actualidle)*100}
        \edef\tposbgcol{Blue!\pgfmathresult!white}
        \pgfmathparse{\pgfmathresult < 50}
        \ifnum0=\pgfmathresult\relax
           \def\tposcol{white}
        \else  
           \def\tposcol{black}
        \fi
        
        \pgfmathparse{(#2/\actualidle)*100}
        \edef\fnegbgcol{Blue!\pgfmathresult!white}
        \pgfmathparse{\pgfmathresult < 50}
        \ifnum0=\pgfmathresult\relax
           \def\fnegcol{white}
        \else  
           \def\fnegcol{black}
        \fi
        
        \pgfmathparse{(#3/\actualspeech)*100}
        \edef\fposbgcol{Blue!\pgfmathresult!white}
        \pgfmathparse{\pgfmathresult < 50}
        \ifnum0=\pgfmathresult\relax
           \def\fposcol{white}
        \else  
           \def\fposcol{black}
        \fi
        
        \pgfmathparse{(#4/\actualspeech)*100}
        \edef\tnegbgcol{Blue!\pgfmathresult!white}
        \pgfmathparse{\pgfmathresult < 50}
        \ifnum0=\pgfmathresult\relax
           \def\tnegcol{white}
        \else  
           \def\tnegcol{black}
        \fi
        
        \matrix (conmat) [row sep=.1cm,column sep=.1cm,ampersand replacement=\&] {
            \node (tpos) [box,
                label={[rotate=90,anchor=center,text depth=1em,yshift=.1cm]left:\( \mathbf{idle} \)},
                label=above:\( \mathbf{idle} \),
                text=\tposcol, fill=\tposbgcol] {#1};
            \&
            \node (fneg) [box,
                label={[yshift=-.1cm]above:\textbf{speech}},
                label=above right:\textbf{total},
                label=right:$\actualidle$,
                text=\fnegcol, fill=\fnegbgcol] {#2};
            \\
            \node (fpos) [box,
                label={[rotate=90,anchor=center,text depth=1em,yshift=.1cm]left:\( \mathbf{speech} \)},
                label=below left:\textbf{total},
                label=below:$\predidle$,
                text=\fposcol, fill=\fposbgcol] {#3};
            \&
            \node (tneg) [box,
                label=right:$\actualspeech$,
                label=below:$\predspeech$,
                text=\tnegcol, fill=\tnegbgcol] {#4};
            \\
        };
        \node [rotate=90,left=.05cm of conmat,anchor=center,align=center] {\textbf{valor real}};
        \node [above=.05cm of conmat] {\textbf{resultado da classificação}};
    \end{tikzpicture}
 }
 
\MakeConfusion{855}{93}{185}{847}