- Perda de precisao ao computar potencias muito elevadas. Mais evidente nos
  blocos com a probabilidade complementar (verificar  np.spy() na matriz final)

- Calculo simbolico para entender o comportamento dos blocos da matriz.

- Encontrar operacao em bloco para reduzir o uso de memoria na computacao.

- encontrar nomeclaturas para os blocos que nao sao referenciados pela
  literatura classica

- Encontrar significado matematico dos blocos nao referenciados na literatura
  classica

- Fazer uma analisa assintotica (provavelmente cheia) da esparsidade da matriz
  de transicao. Entender em qual momento ela torna-se nao esparsa. DEfinir mais
prcisamente o que eh esparsidade de uma matriz.

- Encontrar as funcoes geradoras dos elem,entos das matrizes bloco e suas
  eventuais expansoes. Tentar encontrar um comportamento que permita formular
uma funcao geradora para os elementos. (evitando assim o calculo dos produtos
matriciais).

- Verificar comportamentamento de cada bloco da matriz com relacao a
  esparsividade e melhor algoritmo para executar operacoes.

- Verificar formas de paralelizar a computacao (provavelment utilizando
  blocos).

- Procurar um algoritmo que agrupe de forma mais rapida os resultados ja
  calculados da matriz de transicao (ideia de associatividade). Utilizar
memorizacao e recorrencia (computacional). Fatorizacao por primos?

- Verificar se ha otimizacoes possiveis para o calculo de coeficientes da
  matriz ja calculados em outras partes da matriz. (pouco provavel que haja
ganho significativo).

