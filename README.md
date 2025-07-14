# K-Nearest Neighbors para OCR
Utilizamos neste trabalho o KNN para o reconhecimento de caracteres do alfabeto Iorubá.

## Participantes
Artur Massaro Cremonez <br>
Bruno Henrique Silva Viola

# O que foi feito?
Iniciamos pelo pré-processamento, realizando a conversão das imagens para a escala de cinza. Construímos a possibilidade do recorte das bordas brancas dos caracteres (essa opção deve ser ativada por uma flag). Por fim, o redimensionamento das imagens é executado (por padrão, 32x32. No
## Estrutura do repositório
- `matrizes_de_confusao`: 
- `metricas_por_classe`:
- `relatorios_classificacao`:


















> Pré-processamento
- Realizamos a conversão da imagem para a escala de cinza.
- Possibilidades:
- - Adequação de cores
- - Adequação de dimensões
- - Balancear exemplos entre as classes
- - Utilizar estruturas específicas para organizar os exemplos

> Gerar modelo

- Implementar etapas de treinamento e teste com pelo menos 10 iterações
- Definir o método que irá utilizar
- - Sugestão: começar com modelos simples ou se basear em algum trabalho da literatura
- Definir condição de parada para etapa de treinamento
- - Salvar as métricas usadas
- Classificar o conjunto de teste, calcular os valores das métricas
- - Usar pelo menos 4 métricas a seguir:
- - - Acurácia
- - - Precisão
- - - Revocação
- - - F1-Score

> Pós-processamento

- Caso tenha sido usado alguma transformação nas imagens e seja necessário desfazer para mensurar corretamente as métricas, implementar a restauração após a classificação (nas etapas de treinamento e teste)
