# Build PDFs from my .typ (Typst) files
#
# Requirements:
# * Typst, see [the installation
#   instructions](https://github.com/typst/typst?tab=readme-ov-file#installation)
# * Libertinus fonts, from https://github.com/alerque/libertinus

.PHONY: pdf
pdf: slides

.PHONY: slides
slides:
	typst compile -f pdf slides.typ
	open slides.pdf

.PHONY: slides-long
slides-long:
	typst compile -f pdf slides-long.typ

# Doesn't work for slides, as
# "error: page configuration is not allowed inside of containers"
# as in:
#   #set page(paper: "presentation-16-9")
.PHONY: html
html:
	typst compile --features html -f html slides.typ

.PHONY: clean
clean:
	rm -f slides.pdf slides-long.pdf

.PHONY: help
help:
	@echo "make pdf           alias for `make slides`"
	@echo "make slides        create slides.pdf from slides.typ and open"
	@echo "make slides-long   create slides-long.pdf from slides-long.typ"
	@echo "make clean         delete slides.pdf and slides-long.pdf"
	@echo "make help          gives this help text"
