# OCR with KNN

## 3º Atividade - Gerar modelo de Aprendizado de Máquina para classificar as imagens

> Pré-processamento

- Adequar e organizar as imagens para estruturar os conjuntos de `treinamento` e `teste`.
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
