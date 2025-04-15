from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-2.5-7B")

# Собираем все подозрительные слои (линейные и конволюционные)
quant_layers = []
for name, layer in model.named_modules():
    if isinstance(layer, (torch.nn.Linear, torch.nn.Conv2d)):  # + другие квантуемые типы
        quant_layers.append((name, layer))

print(f"Найдено {len(quant_layers)} квантуемых слоёв:")
for name, _ in quant_layers[:5]:  # Выводим первые 5 для примера
    print(name)


''' Пример вывода:
model.layers.0.self_attn.q_proj  
model.layers.0.self_attn.k_proj  
model.layers.0.self_attn.v_proj  
model.layers.0.self_attn.o_proj  
model.layers.0.mlp.gate_proj 
'''


## Создадим класс для отслеживания LIM всех слоёв:

class LIMTracker:
    def __init__(self, model):
        self.model = model
        self.lim_data = {}
        self.hooks = []
        
        # Регистрируем хуки на все линейные слои
        for name, layer in model.named_modules():
            if isinstance(layer, (torch.nn.Linear, torch.nn.Conv2d)):
                hook = layer.register_forward_hook(self._save_io(name))
                self.hooks.append(hook)
    
    def _save_io(self, layer_name):
        def hook(module, input, output):
            self.lim_data[layer_name] = {
                'input': input[0].detach().clone(),
                'output': output.detach().clone()
            }
        return hook
    
    def compute_lim(self):
        lim_results = {}
        for name, data in self.lim_data.items():
            input_norm = torch.norm(data['input'], p=2)
            diff_norm = torch.norm(data['output'] - data['input'], p=2)
            lim_results[name] = (diff_norm / input_norm).item()
        return lim_results
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

# Инициализация
lim_tracker = LIMTracker(model)



## Модифицируем run_calib_mode для сбора LIM:


def run_calib_mode(self):
    lim_tracker = LIMTracker(self.model)  # Инициализация трекера
    lim_values = {name: [] for name in lim_tracker.lim_data.keys()}  # Словарь для сбора метрик
    
    for data in tqdm(self.calib_data, desc="Calibration"):
        if isinstance(data, (tuple, list)):
            self.model(*data)  # Обычный forward
        else:
            self.model(data)
        
        # Собираем LIM после каждого батча
        current_lim = lim_tracker.compute_lim()
        for name, val in current_lim.items():
            lim_values[name].append(val)
    
    # Усредняем LIM по всем батчам
    avg_lim = {name: sum(vals)/len(vals) for name, vals in lim_values.items()}
    
    # Выводим топ-5 слоёв с наибольшим LIM
    print("Топ-5 слоёв по LIM:")
    for name, lim in sorted(avg_lim.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"{name}: {lim:.4f}")
    
    lim_tracker.remove_hooks()  # Очистка хуков

'''Пример вывода

Calibration: 100%|█████████| 100/100 [01:23<00:00]  
Топ-5 слоёв по LIM:  
model.layers.15.self_attn.o_proj: 0.1421  
model.layers.23.mlp.down_proj: 0.1317  
model.layers.5.self_attn.v_proj: 0.1212  
model.layers.0.mlp.gate_proj: 0.1123  
model.layers.30.self_attn.q_proj: 0.1038  
'''

''' Как интерпретировать LIM?
  LIM ≈ 0: Слой почти не искажает входные данные (например, LayerNorm).
  LIM > 0.1: Значительные изменения (возможно, квантование вносит ошибки).
  LIM > 0.5: Критичные искажения (требуется проверить битность квантования).

Важно!
 - Для Qwen-2.5 уточните точные имена слоёв через print(model).
 - Если модель уже квантована — LIM покажет фактическое искажение данных.
 - Для статического квантования считайте LIM после калибровки.
'''
