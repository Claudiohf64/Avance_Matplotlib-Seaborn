import java.util.Arrays;
import java.util.Iterator;

public class Ejercicio5 {
	static void verArray(int[][] valores, int i) {
		if (i==valores.length) {
			return;
		}
		System.out.println(valores[i]);
		verArray(valores, i+1);
	}
	
	static void verArray2(int[] valores, int i) {
		int suma = 0;
		if (i==valores.length) {
			return;
		}
		for (int j = 0; j < valores.length; j++) {
			suma += valores[j];
		}
		System.out.println(suma);
	}
	
	public static void main(String[] args) {
		
		int[] valores= {1,2,3,4,5};
		verArray2(valores, 0);
	}
	

}
