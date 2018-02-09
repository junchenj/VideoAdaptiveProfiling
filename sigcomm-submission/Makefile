# sane targets
# DOCN = outline
# DOCN = cfa-nsdi
# DOCN = main-nsdi-ddn
DOCN = main

all: doc 
doc:
	pdflatex $(DOCN).tex
	pdflatex $(DOCN).tex
	bibtex $(DOCN) || echo "ok"
	pdflatex $(DOCN).tex
	pdflatex $(DOCN).tex
	#cp $(DOCN).pdf /media/Common/
pdf: ps
	ps2pdf14 -dEmbedAllFonts=true $(DOCN).ps
ps:  doc	
	#dvips  -Ppdf -K -Pdownload35 -Pcmz -Pamz -D 600 -G0 -t letter  $(DOCN).dvi -o $(DOCN).ps
	dvips  -Ppdf -Pcmz -Pamz -t letter -D 600 -G1  $(DOCN).dvi -o $(DOCN).ps

clean:
	rm -f *.aux *.dvi *.idx *.ind *.log *.toc *.bak *.blg *.bbl
	rm $(DOCN).pdf
distclean: clean
	rm -f *~

# Tell versions [3.59,3.63) of GNU make to not export all variables.
# Otherwise a system limit (for SysV at least) may be exceeded.
.NOEXPORT:
