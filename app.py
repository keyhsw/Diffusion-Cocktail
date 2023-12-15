import os

import gradio as gr
import numpy as np
import torch
import torchvision.transforms as T

from clip_interrogator import Config, Interrogator
from diffusers import StableDiffusionPipeline

from ditail import DitailDemo, seed_everything

BASE_MODEL = {
    'sd1.5': 'runwayml/stable-diffusion-v1-5',
    # 'sd1.5': './ditail/model/stable-diffusion-v1-5'
    'realistic vision': 'stablediffusionapi/realistic-vision-v51',
    'pastel mix (anime)': 'stablediffusionapi/pastel-mix-stylized-anime',
    'chaos (abstract)': 'MAPS-research/Chaos3.0',
}

# LoRA trigger words
LORA_TRIGGER_WORD = {
    'none': [],
    'film': ['film overlay', 'film grain'],
    'snow': ['snow'],
    'flat': ['sdh', 'flat illustration'],
    'minecraft': ['minecraft square style', 'cg, computer graphics'],
    'animeoutline': ['lineart', 'monochrome'],
    # 'caravaggio': ['oil painting', 'in the style of caravaggio'],
    'impressionism': ['impressionist', 'in the style of Monet'],
    'pop': ['POP ART'],
    'shinkai_makoto': ['shinkai makoto', 'kimi no na wa.', 'tenki no ko', 'kotonoha no niwa'],
}


class WebApp():
    def __init__(self, debug_mode=False):
        self.args_base = {
            "seed": 42,
            "device": "cuda",
            "output_dir": "output_demo",
            "caption_model_name": "blip-large",
            "clip_model_name": "ViT-L-14/openai",
            "inv_model": "stablediffusionapi/realistic-vision-v51",
            "spl_model": "runwayml/stable-diffusion-v1-5",
            "inv_steps": 50,
            "spl_steps": 50,
            "img": None,
            "pos_prompt": '',
            "neg_prompt": 'worst quality, blurry, NSFW',
            "alpha": 3.0,
            "beta": 0.5,
            "omega": 15,
            "mask": None,
            "lora": "none",
            "lora_dir": "./ditail/lora",
            "lora_scale": 0.7,
            "no_injection": False,
        }

        self.args_input = {} # for gr.components only
        self.gr_loras = list(LORA_TRIGGER_WORD.keys())

        self.gtag = os.environ.get('GTag')

        self.ga_script = f"""
            <script async src="https://www.googletagmanager.com/gtag/js?id={self.gtag}"></script>
            """
        self.ga_load = f"""
            function() {{
                window.dataLayer = window.dataLayer || [];
                function gtag(){{dataLayer.push(arguments);}}
                gtag('js', new Date());

                gtag('config', '{self.gtag}');
            }}
            """
        
        # pre-download base model for better user experience
        self._preload_pipeline()

        self.debug_mode = debug_mode # turn off clip interrogator when debugging for faster building speed
        if not self.debug_mode:
            self.init_interrogator()


    def init_interrogator(self):
        config = Config()
        config.clip_model_name = self.args_base['clip_model_name']
        config.caption_model_name = self.args_base['caption_model_name']
        self.ci = Interrogator(config)
        self.ci.config.chunk_size = 2048 if self.ci.config.clip_model_name == "ViT-L-14/openai" else 1024
        self.ci.config.flavor_intermediate_count = 2048 if self.ci.config.clip_model_name == "ViT-L-14/openai" else 1024


    def _preload_pipeline(self):
        for model in BASE_MODEL.values():
            pipe = StableDiffusionPipeline.from_pretrained(
                model, torch_dtype=torch.float16
            ).to(self.args_base['device'])
        pipe = None


    def title(self):
        gr.HTML(
                """
                <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
                <div>
                    <h1 >Diffusion Cocktail 🍸: Fused Generation from Diffusion Models</h1>
                    <div style="display: flex; justify-content: center; align-items: center; text-align: center; margin: 20px; gap: 10px;">
                        <a class="flex-item" href="https://arxiv.org/abs/2312.08873" target="_blank">
                            <img src="https://img.shields.io/badge/arXiv-paper-darkred.svg" alt="arXiv Paper">
                        </a>                      
                        <a class="flex-item" href="https://MAPS-research.github.io/Ditail" target="_blank">
                            <img src="https://img.shields.io/badge/Project_Page-Diffusion_Cocktail-yellow.svg" alt="Project Page">
                        </a>
                        <a class="flex-item" href="https://github.com/MAPS-research/Ditail" target="_blank">
                            <img src="https://img.shields.io/badge/Github-Code-green.svg" alt="GitHub Code">
                        </a>
                    </div>
                </div>
                </div>
                """
                )
                

    def get_image(self):
        self.args_input['img'] = gr.Image(label='content image', type='pil', show_share_button=False, elem_classes="input_image")
    
    def get_prompts(self):
        # with gr.Row():
        generate_prompt = gr.Checkbox(label='generate prompt with clip', value=True)
        self.args_input['pos_prompt'] = gr.Textbox(label='prompt')
            
            
        # event listeners
        self.args_input['img'].upload(self._interrogate_image, inputs=[self.args_input['img'], generate_prompt], outputs=[self.args_input['pos_prompt']])
        generate_prompt.change(self._interrogate_image, inputs=[self.args_input['img'], generate_prompt], outputs=[self.args_input['pos_prompt']])


    def _interrogate_image(self, image, generate_prompt):
        # self.init_interrogator()
        if hasattr(self, 'ci') and generate_prompt:
            return self.ci.interrogate_fast(image).split(',')[0].replace('arafed', '')
        else:
            return ''
        

    def get_base_model(self):
        self.args_input['spl_model'] = gr.Radio(choices=list(BASE_MODEL.keys()), value=list(BASE_MODEL.keys())[0], label='target base model')

    def get_lora(self, num_cols=3):
        self.args_input['lora'] = gr.State('none')
        lora_gallery = gr.Gallery(label='target LoRA (optional)', columns=num_cols, value=[(os.path.join(self.args_base['lora_dir'], f"{lora}.jpeg"), lora) for lora in self.gr_loras], allow_preview=False, show_share_button=False, selected_index=0)
        lora_gallery.select(self._update_lora_selection, inputs=[], outputs=[self.args_input['lora']])
    
    def _update_lora_selection(self, selected_state: gr.SelectData):
        return self.gr_loras[selected_state.index]

    def get_params(self):
        with gr.Row():
            with gr.Column():
                self.args_input['inv_model'] = gr.Radio(choices=list(BASE_MODEL.keys()), value=list(BASE_MODEL.keys())[1], label='inversion base model')
                self.args_input['neg_prompt'] = gr.Textbox(label='negative prompt', value=self.args_base['neg_prompt'])
            # with gr.Row():
                self.args_input['alpha'] = gr.Number(label='positive prompt scaling weight (alpha)', value=self.args_base['alpha'], interactive=True)
                self.args_input['beta'] = gr.Number(label='negative prompt scaling weight (beta)', value=self.args_base['beta'], interactive=True)

            with gr.Column():
                self.args_input['omega'] = gr.Slider(label='cfg', value=self.args_base['omega'], maximum=25, interactive=True)
                
                self.args_input['inv_steps'] = gr.Slider(minimum=1, maximum=100, label='edit steps', interactive=True, value=self.args_base['inv_steps'], step=1)
                self.args_input['spl_steps'] = gr.Slider(minimum=1, maximum=100, label='sample steps', interactive=False, value=self.args_base['spl_steps'], step=1, visible=False)
                # sync inv_steps with spl_steps
                self.args_input['inv_steps'].change(lambda x: x, inputs=self.args_input['inv_steps'], outputs=self.args_input['spl_steps'])

                self.args_input['lora_scale'] = gr.Slider(minimum=0, maximum=1, label='LoRA scale', value=0.7)
                self.args_input['seed'] = gr.Number(label='seed', value=self.args_base['seed'], interactive=True, precision=0, step=1)

    def run_ditail(self, *values):
        self.args = self.args_base.copy()
        print(self.args_input.keys())
        for k, v in zip(list(self.args_input.keys()), values):
            self.args[k] = v
        # quick fix for example
        self.args['lora'] = 'none' if not isinstance(self.args['lora'], str) else self.args['lora']
        print('selected lora: ', self.args['lora'])
        # map inversion model to url
        self.args['pos_prompt'] = ', '.join(LORA_TRIGGER_WORD.get(self.args['lora'], [])+[self.args['pos_prompt']])
        self.args['inv_model'] = BASE_MODEL[self.args['inv_model']]
        self.args['spl_model'] = BASE_MODEL[self.args['spl_model']]
        print('selected model: ', self.args['inv_model'], self.args['spl_model'])

        seed_everything(self.args['seed'])
        ditail = DitailDemo(self.args)
        
        metadata_to_show = ['inv_model', 'spl_model', 'lora', 'lora_scale', 'inv_steps', 'spl_steps', 'pos_prompt', 'alpha', 'neg_prompt', 'beta', 'omega']
        self.args_to_show = {}
        for key in metadata_to_show:
            self.args_to_show[key] = self.args[key ]

        return ditail.run_ditail(), self.args_to_show
        # return self.args['img'], self.args

    def run_example(self, img, prompt, inv_model, spl_model, lora):
        return self.run_ditail(img, prompt, spl_model, gr.State(lora), inv_model)

    def show_credits(self):
        # gr.Markdown(
        #     """
        #     ### About Diffusion Cocktail (Ditail)
        #     * This is a research project by [MAPS Lab](https://whongyi.github.io/MAPS-research), [NYU Shanghai](https://shanghai.nyu.edu)
        #     * Authors: Haoming Liu (haoming.liu@nyu.edu), Yuanhe Guo (yuanhe.guo@nyu.edu), Hongyi Wen (hongyi.wen@nyu.edu)
        #     """
        # )
        gr.Markdown(
            """
            ### Model Credits
            * Diffusion Models are downloaded from [huggingface](https://huggingface.co) and [civitai](https://civitai.com): [stable diffusion 1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5), [realistic vision](https://huggingface.co/stablediffusionapi/realistic-vision-v51), [pastel mix](https://huggingface.co/stablediffusionapi/pastel-mix-stylized-anime), [chaos3.0](https://civitai.com/models/91534/chaos30)
            * LoRA Models are downloaded from [civitai](https://civitai.com) and [liblib](https://www.liblib.art): [film](https://civitai.com/models/90393/japan-vibes-film-color), [snow](https://www.liblib.art/modelinfo/f732b23b02f041bdb7f8f3f8a256ca8b), [flat](https://www.liblib.art/modelinfo/76dcb8b59d814960b0244849f2747a15), [minecraft](https://civitai.com/models/113741/minecraft-square-style), [animeoutline](https://civitai.com/models/16014/anime-lineart-manga-like-style), [impressionism](https://civitai.com/models/113383/y5-impressionism-style), [pop](https://civitai.com/models/161450?modelVersionId=188417), [shinkai_makoto](https://civitai.com/models/10626?modelVersionId=12610) 
            """
        )


    def ui(self):
        with gr.Blocks(css='.input_image img {object-fit: contain;}', head=self.ga_script) as demo:
            self.title()
            with gr.Row():
                # with gr.Column():
                self.get_image()

                with gr.Column():
                    self.get_prompts()
                    self.get_base_model()
                    self.get_lora(num_cols=3)
                    submit_btn = gr.Button("Generate", variant='primary')

            with gr.Accordion("advanced options", open=False):
                self.get_params()   
            
            with gr.Row():
                output_image = gr.Image(label="output image")
                # expected_output_image = gr.Image(label="expected output image", visible=False)
                metadata = gr.JSON(label='metadata')

                submit_btn.click(self.run_ditail,
                                inputs=list(self.args_input.values()),
                                outputs=[output_image, metadata],
                                scroll_to_output=True,
                                )

            with gr.Row():
                cache_examples = not self.debug_mode
                gr.Examples(
                    examples=[[os.path.join(os.path.dirname(__file__), "example", "Lenna.png"), 'a woman called Lenna wearing a feathered hat', list(BASE_MODEL.keys())[1], list(BASE_MODEL.keys())[2], 'none']],
                    inputs=[self.args_input['img'], self.args_input['pos_prompt'], self.args_input['inv_model'], self.args_input['spl_model'], gr.Textbox(label='LoRA', visible=False), ],
                    fn = self.run_example,
                    outputs=[output_image, metadata],
                    run_on_click=True,
                    cache_examples=cache_examples,
                )

            self.show_credits()
        
            demo.load(None, js=self.ga_load)
        return demo


app = WebApp(debug_mode=False)
demo = app.ui()


if __name__ == "__main__":
    demo.launch(share=True)
    # demo.launch()
    
    