from calculator import Calculator


def demo():
	calc = Calculator()
	a, b = 12, 4
	print("Demo operations:")
	print(f"{a} + {b} =", calc.add(a, b))
	print(f"{a} - {b} =", calc.sub(a, b))
	print(f"{a} * {b} =", calc.mul(a, b))
	print(f"{a} / {b} =", calc.div(a, b))
	print(f"{a} ** {b} =", calc.pow(a, b))
	print(f"{a} % {b} =", calc.mod(a, b))


def repl():
	calc = Calculator()
	print("\nInteractive mode. Commands: add, sub, mul, div, pow, mod, quit")
	while True:
		try:
			line = input('> ').strip()
		except (EOFError, KeyboardInterrupt):
			print('\nExiting.')
			break
		if not line:
			continue
		if line.lower() in ('q', 'quit', 'exit'):
			print('Goodbye.')
			break
		parts = line.split()
		if len(parts) < 3:
			print('Usage: <op> <num1> <num2>')
			continue
		op, x_s, y_s = parts[0], parts[1], parts[2]
		try:
			x = float(x_s) if ('.' in x_s) else int(x_s)
			y = float(y_s) if ('.' in y_s) else int(y_s)
		except ValueError:
			print('Invalid numbers')
			continue
		try:
			if op in ('add', '+'):
				res = calc.add(x, y)
			elif op in ('sub', '-'):
				res = calc.sub(x, y)
			elif op in ('mul', '*'):
				res = calc.mul(x, y)
			elif op in ('div', '/'):
				res = calc.div(x, y)
			elif op in ('pow', '^'):
				res = calc.pow(x, y)
			elif op in ('mod', '%'):
				res = calc.mod(x, y)
			else:
				print('Unknown operation')
				continue
			print(res)
		except Exception as e:
			print('Error:', e)


if __name__ == '__main__':
	demo()
	repl()