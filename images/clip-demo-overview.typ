// A standalone version of the CLIP demo overview diagram from the slides at
// https://github.com/Aiven-Labs/app-multimodal-search-CLIP-PostgreSQL/blob/main/slides/slides.typ
//
// Create an output SVG file by installing typst (see https://typst.app/docs/)
// and running
//
//    typst compile -f svg clip-demo-overview.typ

// Use Fletcher for diagrams
#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node
#import fletcher.shapes: chevron, pill

// Make the page fit the diagram
#set page(width: auto, height: auto, margin: 0pt)

// Our brand guidelines say to use Inter for general text
// Check if it's available at the command line:
//   typst fonts | grep Inter
#set text(font: "Inter")

// I don't want "Figure 1:" in the figure caption text
#show figure.caption: it => [
  #it.body
]

#set text(size: 20pt)

#diagram(
  // The default spacing between rows and columns is 3em, which is a bit
  // big for a slide, especially vertically with 3 rows
  spacing: (1.2em, 0.3em),

  // For debugging placement, it's useful to see the actual node
  //node-fill: teal.lighten(50%),

  // By default, nodes are rectangular or circular depending on their aspect
  // ratio. I want more control than that, so will make all nodes rectangular
  node-shape: rect,

  // Let's have a bit more gap between a node and its edge(s)
  node-outset: 5pt,

  node((0, 0), name: <photos>, figure(
    image("unsplash-dog-photo.png", width: 5em),
    caption: text(size: 20pt)[Photos from\ Unsplash],
  )),

  node(
    (2, 0),
    name: <clip-top>,
    figure(
      image("openai-clip.png", width: 3em),
      caption: text(size: 18pt)[CLIP model\ from OpenAI],
    ),
  ),

  node((4, 0), name: <vectors>, [Vectors in 512\ dimension space]),

  node((5, 2), name: <postgres>, grid(
    columns: (auto, auto),
    gutter: 0.5em,
    image("elephant.png", width: 3em), text(size: 20pt)[PostgreSQL],
  )),

  node((0, 4), name: <search>, image("search-phrase.png", width: 8em)),

  node((2, 4), name: <clip-bottom>, figure(
    image("openai-clip.png", width: 3em),
    caption: text(size: 18pt)[CLIP model\ from OpenAI],
  )),

  //node((4, 4), name: <single-vector>, [Single vector in 512\ dimension space]),
  node((4, 4), name: <single-vector>, [Vector in 512\ dimension space]),

  edge(<photos>, "->", <clip-top>),
  edge(<clip-top>, "->", <vectors>),
  edge(<vectors>, "->", <postgres>),

  edge(<search>, "->", <clip-bottom>),
  edge(<clip-bottom>, "->", <single-vector>),
  edge(<single-vector>, "->", <postgres>),
)
