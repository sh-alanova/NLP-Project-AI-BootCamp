import torch
from tqdm import tqdm

class LIMTracker:
    def __init__(self, model, target_layer_types=(torch.nn.Linear,)):
        self.model = model
        self.lim_data = {}  # {layer_name: {'input': [], 'output': []}}
        self.hooks = []
        
        # Регистрируем хуки на все целевые слои
        for name, layer in model.named_modules():
            if isinstance(layer, target_layer_types):
                self.lim_data[name] = {'input': None, 'output': None}
                self.hooks.append(layer.register_forward_hook(self._create_hook(name)))

    def _create_hook(self, layer_name):
        def hook(module, input, output):
            self.lim_data[layer_name]['input'] = input[0].detach().clone()
            self.lim_data[layer_name]['output'] = output.detach().clone()
        return hook

    def compute_lim(self):
        """Возвращает словарь {layer_name: lim_value}"""
        lim_results = {}
        for name, data in self.lim_data.items():
            if data['input'] is not None and data['output'] is not None:
                lim = torch.norm(data['output'] - data['input']) / torch.norm(data['input'])
                lim_results[name] = lim.item()
        return lim_results

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()



### Main

def main():
    # 1. Инициализация модели и данных (ваш код)
    model = ...  # Загружаем квантованную модель Qwen-2.5
    calib_data = ...  # Ваши калибровочные данные

    # 2. Инициализация трекера LIM
    # Указываем типы слоёв, которые хотим отслеживать (напр., LinearQuantizer)
    lim_tracker = LIMTracker(model, target_layer_types=(LinearQuantizer,))

    # 3. Прогон данных через модель
    with torch.no_grad():
        for batch in tqdm(calib_data, desc="Вычисление LIM"):
            if isinstance(batch, (tuple, list)):
                model(*batch)
            else:
                model(batch)

    # 4. Получение и сохранение результатов
    lim_results = lim_tracker.compute_lim()
    
    print("\nРезультаты LIM:")
    for name, lim in lim_results.items():
        print(f"{name}: {lim:.4f}")

    # 5. Очистка хуков (обязательно!)
    lim_tracker.remove_hooks()

    # (Опционально) Сохранение в файл
    import json
    with open("lim_results.json", "w") as f:
        json.dump(lim_results, f, indent=2)

  ```
🔹 Дополнительные возможности
 - Фильтрация слоёв: Если нужно отслеживать только конкретные слои (например, с квантованием активаций):

  target_layers = [name for name, layer in model.named_modules() 
                  if isinstance(layer, LinearQuantizer) and layer.quant_input.is_enable]
  lim_tracker = LIMTracker(model, target_layer_names=target_layers)


 - Визуализация

  import matplotlib.pyplot as plt
  
  def plot_lim(lim_results):
      names = list(lim_results.keys())
      values = list(lim_results.values())
      
      plt.figure(figsize=(12, 6))
      plt.bar(range(len(values)), values, tick_label=names)
      plt.xticks(rotation=90)
      plt.ylabel("LIM")
      plt.title("Искажение при квантовании по слоям")
      plt.tight_layout()
      plt.show()
  
  plot_lim(lim_results)


🔹 Важные замечания
 - Производительность: Хуки добавляют оверхед (~5-10% slowdown).
 - Память: Сохранение тензоров может требовать много памяти (используйте .cpu()).
 - Для Qwen-2.5: Особое внимание к LinearQuantizer и Conv2dQuantizer.
```
