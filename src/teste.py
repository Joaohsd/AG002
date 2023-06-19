def mapear_faixa_idade(idade):
    faixa = (idade // 10)
    return str(faixa) if 1 <= faixa <= 9 else 'Faixa de idade inválida'

# Exemplo de uso
idade = int(input('Digite a idade: '))
faixa = mapear_faixa_idade(idade)
print(faixa)
