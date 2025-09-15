# Notes

History:

- This talk is based on the workshop in the parent directory of this repository.
- That workshop is in turn based on the OpenSearch original, at
  https://github.com/Aiven-Labs/workshop-multimodal-search-CLIP-OpenSearch
- And that is inspired by an article in the Aiven developer center, using
  OpenSearch and ~25,000 Unsplash images
  https://aiven.io/developer/opensearch-multimodal-search
  (this URL may change, but should redirect to the new location)

----

Introduction to CLIP
https://openai.com/index/clip/ 

Also reference to Llava https://llava-vl.github.io/ which relies on CLIP

----

openAI CLIP dates from 2021, so well established

https://github.com/openai/CLIP
> CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on
> a variety of (image, text) pairs. It can be instructed in natural language to
> predict the most relevant text snippet, given an image, without directly
> optimizing for the task, similarly to the zero-shot capabilities of GPT-2 and 3.

Texts are encoded

Images are encoded

Those encodings are then aligned in a shared embedding space, enabling the
model to perform tasks that require understanding the relationship between 
images and  text.
This does mean it’s only going to do really well with text that falls within the
text training sets.
And of course all the normal caveats about bias, inclusivity, data range and so
on always apply.

Apparently trained on 400 million image-text pairs collected from the Internet.

Specifically, the contrastive approach means:
- It's presented with paired image and text - those represent as close as 
  posible in the embedding space
- The model is incentivised to make unpaired inputs ("a photo of a car" and 
  a picture of a dog) represent as far apart as possible
And CLIP is used as the basis for other multimodal applications.

**ViT-B/32**:  uses a Vision Transformer (ViT) as the image encoder, with an
image patch size of 32 (larger patch size means fewer patches per image at a
given size, and thus faster/smaller models) and the ‘Base’ size model. Larger
models with smaller patch sizes will tend to need more memory and compute in
exchange for better performance

The openAI CLIP GitHub repository: https://github.com/openai/CLIP gives example
Python code similar to what we’re doing (but without the database storage). It
also shows how to do the reverse - take an image and return its
prediction/description of what the image contains.

`clip.load` returns
* `model`
* `preprocess` - the TorchVision transform needed by the model

`photos_preprocessed = torch.stack([preprocess(photo) for photo in photos]).to(device)`
* `preprocess` was returned by `clip.load`
* `torch.stack(...)` concatenates the array of results
* `.to(device)` …

`torch.nograd` disables gradient calculation, whih we don't need (reducing
memory consumption)
* `model.encode_image` takes the concatenated data
* `a /= b` is (essentially) short for `a = a / b`
* `photos_features /= photos_features.norm(dim=-1, keepdim=True)` normalises
  all the vectors so they’re the same “length”, which means we can compare
  distance between vectors

The openAI CLIP GitHub README also suggests looking at
* [OpenCLIP](https://github.com/mlfoundations/open_clip): "includes larger 
  and independently trained CLIP models up to ViT-G/14"
* [Hugging Face implementation of CLIP](https://huggingface.co/docs/transformers/model_doc/clip):
  "for easier integration with the HF ecosystem"

Example prompts
* cat; dog on chair;
* man jumping; 
* snowy streets; woolen hat; santa claus _– because it finds christmas 
  scenes?_

More background on multimodal models
* 2023, Chip Huyen, [Multimodality and Large Multimodal Models
  (LMMs)](https://huyenchip.com/2023/10/10/multimodal.html)
* 2024, `[at] Editorial Team` [An Introduction to Large Multimodal Models](https://www.alexanderthamm.com/en/blog/an-introduction-to-large-multimodal-models/)
* 2024, Harry Guinness [How does ChatGPT work?](https://zapier.com/blog/how-does-chatgpt-work/)

and

* 2024 Voxel51 [A History of CLIP Model Training Data Advances](https://voxel51.com/blog/a-history-of-clip-model-training-data-advances)

## Note on images in the `photos/` folder


The images in the `photos/` directory came from Unsplash and have been reduced in size
to make them fit within GitHub repository filesize limits.

When the app is running and shows images, it uses the URL for the image in the
GitHub repository in the `<img>` tag.

This is not good practice for a production app, as GitHub is not intended to act
as an image repository for web apps.

We should instead store the images in a proper filestore, or even in a database
(PostgreSQL or perhaps Valkey). But that would almost certainly require us to
retrieve the image to local storage display, and we'd then need to make sure
we didn't _keep_ them too long, as we don't have much local storage. This is
the sort of design discussion that we'd need for anything beyond this MVP
(minimum viable product).
