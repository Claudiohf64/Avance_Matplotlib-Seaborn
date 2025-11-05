
public class Ejercicio4 {
	
	static int sumarHastaRecursivo(int numero) {
		if (numero== 0) {
			return 0;
		}
		int suma = numero + sumarHastaRecursivo(numero-1);
		return suma;
		
	}
	
	static int sumarHastaRecursivo2(int numero) {
		if (numero== 1) {
			return 1;
		}
		int fact = numero * sumarHastaRecursivo2(numero-1);
		return fact;
		
	}

	static int linea(int numero) {
		if (numero== 0) {
			return 0;
		}
		linea(numero-1);
		System.out.print("*");
		return numero;
	}
	
	static int linea2(int numero) {
		if (numero== 0) {
			return 0;
		}
		linea(numero-1);
		System.out.print("*");
		return numero;
	}
	
	static void mostrarArray(int [] valores) {
		for (int i = 0; i < valores.length; i++) {
			System.out.println(valores	[i]);
		}
	}
	
	public static void main(String[] args) {
		System.out.println(sumarHastaRecursivo(5));
		System.out.println(sumarHastaRecursivo2(5));
		linea(5);
		mostrarArray(new int[] {1,2,32,4,5,6,7,22,33,14});
	}

}
