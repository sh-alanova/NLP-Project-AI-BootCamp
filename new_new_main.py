class LIMTracker:
    def __init__(self, model):
        self.model = model
        self.lim_data = {}
        self.hooks = []
        
        # Регистрируем хуки только на квантованные Linear/Conv слои
        for name, layer in model.named_modules():
            if isinstance(layer, (LinearQuantizer, Conv2dQuantizer)):
                self.lim_data[name] = {'input': None, 'output': None}
                self.hooks.append(layer.register_forward_hook(self._create_hook(name)))

    def _create_hook(self, layer_name):
        def hook(module, input, output):
            # Преобразуем input[0] в float, если это токены (LongTensor)
            if input[0].dtype == torch.long:
                with torch.no_grad():
                    # Получаем эмбеддинги для токенов
                    embeddings = self.model.get_input_embeddings()(input[0])
                    self.lim_data[layer_name]['input'] = embeddings.clone()
            else:
                self.lim_data[layer_name]['input'] = input[0].detach().clone()
            
            self.lim_data[layer_name]['output'] = output.detach().clone()
        return hook

    def compute_lim(self):
        lim_results = {}
        for name, data in self.lim_data.items():
            if data['input'] is not None and data['output'] is not None:
                # Проверка размерностей
                if data['input'].shape != data['output'].shape:
                    print(f"Предупреждение: размеры input и output не совпадают в слое {name}")
                    continue
                lim = torch.norm(data['output'] - data['input']) / torch.norm(data['input'])
                lim_results[name] = lim.item()
        return lim_results

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()



def main():
    model = ...  # Загруженная квантованная Qwen-2.5
    calib_data = ...  # Данные в формате токенов (LongTensor)

    # Инициализация трекера
    lim_tracker = LIMTracker(model)

    # Прогон данных
    with torch.no_grad():
        for batch in tqdm(calib_data):
            if isinstance(batch, (tuple, list)):
                model(*batch)  # batch = [input_ids, attention_mask, ...]
            else:
                model(batch)   # batch = input_ids

    # Результаты
    lim_results = lim_tracker.compute_lim()
    print("LIM для квантованных слоёв:")
    for name, lim in lim_results.items():
        print(f"{name}: {lim:.4f}")

    lim_tracker.remove_hooks()
