# plotter.py
import json
import matplotlib.pyplot as plt
import config

plt.rcParams['font.family'] ='Malgun Gothic'

class LossPlotter:
    def __init__(self, data_path=config.GRAPH_DIR/'graph_data.json'):
        self.data_path = data_path
        self.data = None

    def load_data(self):
        try:
            with open(self.data_path, 'r') as f:
                self.data = json.load(f)
            print(f"'{self.data_path}'에서 데이터를 성공적으로 불러왔습니다.")
        except FileNotFoundError:
            print(f"error: '{self.data_path}' 파일을 찾을 수 없습니다. 먼저 train.py를 실행하세요.")
            self.data = None

    def plot(self, output_path=config.GRAPH_DIR/'loss_graph.png'):
        if not self.data:
            print("플롯할 데이터가 없습니다.")
            return

        steps = self.data['steps']
        train_losses = self.data['train_losses']
        val_losses = a= self.data['val_losses']

        plt.figure(figsize=(10, 6))
        plt.plot(steps, train_losses, label='훈련 손실(Train Loss)')
        plt.plot(steps, val_losses, label='검증 손실(Validation Loss)')
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Step에 따른 Train Loss와 Validation Loss 변화")
        plt.legend()
        plt.grid(True)
        
        plt.savefig(output_path)
        print(f"손실 그래프가 '{output_path}' 파일로 저장되었습니다.")

if __name__ == "__main__":
    plotter = LossPlotter()
    plotter.load_data()
    plotter.plot()