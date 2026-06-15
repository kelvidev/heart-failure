import train
import inference


def main():
    print("=" * 70)
    print("ETAPA 1 - PRE-PROCESSAMENTO E TREINO DO AGRUPAMENTO")
    print("=" * 70)
    train.run_training()

    print("\n" + "=" * 70)
    print("ETAPA 2 - INFERENCIA EM PACIENTES DESCONHECIDOS")
    print("=" * 70)
    inference.demo()


if __name__ == "__main__":
    main()
