import re

f = open("mode_presentation.tex")


f1 = f.read()
f1 = f1.replace('\documentclass[11pt]{article}','')


preamble = re.split(r'\\begin{document}', f1)[0]

actual_text = re.split(r'\\begin{document}', f1)[1]



actual_text = actual_text.replace('\maketitle','')
actual_text = actual_text.replace('\end{document}','')

actual_text = actual_text.replace('XYZ1',r'\ref{ganarch}')
actual_text = actual_text.replace('XYZ2',r'\ref{minibatch}')
actual_text = actual_text.replace('XYZ3',r'\ref{gen_obj}')



preamble = preamble.replace(r'\def\maxwidth{\ifdim\Gin@nat@width>\linewidth\linewidth',
                            r'%\def\maxwidth{\ifdim\Gin@nat@width>\linewidth\linewidth')
preamble = preamble.replace(r'\else\Gin@nat@width\fi}',r'%\else\Gin@nat@width\fi}')

preamble = preamble.replace(r'\let\Oldincludegraphics\includegraphics',
                            r'%\let\Oldincludegraphics\includegraphics')
preamble = preamble.replace(r'\renewcommand{\includegraphics}[1]{\Oldincludegraphics[width=.8\maxwidth]{#1}}',
                            r'%\renewcommand{\includegraphics}[1]{\Oldincludegraphics[width=.8\maxwidth]{#1}}')
preamble = preamble.replace(r'\geometry{verbose,tmargin=1in,bmargin=1in,lmargin=1in,rmargin=1in}',
                            r'%\geometry{verbose,tmargin=1in,bmargin=1in,lmargin=1in,rmargin=1in}')
preamble = preamble.replace(r'\usepackage{geometry}','\usepackage[margin=1.25in]{geometry}')
preamble = preamble.replace('colorlinks=true','colorlinks=false')
preamble = preamble.replace(r'\DeclareCaptionLabelFormat{nolabel}{}',r'%\DeclareCaptionLabelFormat{nolabel}{}')
preamble = preamble.replace(r'\captionsetup{labelformat=nolabel}',r'%\captionsetup{labelformat=nolabel}')




galr_preamble = open('/Users/Billy/Documents/Uni/cam/GAN/essay tex/mode_preamble.tex', 'w')
galr_preamble.write(preamble)
galr_preamble.close()


galr_text = open('/Users/Billy/Documents/Uni/cam/GAN/essay tex/mode_text.tex', 'w')
galr_text.write(actual_text)
galr_text.close()