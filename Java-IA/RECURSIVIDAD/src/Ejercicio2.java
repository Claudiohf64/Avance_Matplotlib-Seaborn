
public class Ejercicio2 {
	static void decrementarRecursivo(int numero) {
		if (numero == 0) {
			return;
		}
		decrementarRecursivo(numero-1);//MIENTRAS SE APILAN LOS CONTEXTOS RECURSIVOS
		System.out.println(numero);
		//MIENTRAS SE RESUELVEN LOS CONTEXTOS RECURSIVOS
	}
	
	public static void main(String[] args) {
		int i = 0;
		for (int j= 0; j>i;j++) {
		}
		decrementarRecursivo(i);
	}
}


