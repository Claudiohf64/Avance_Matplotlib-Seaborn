public class TestArbol {
	public static void main(String[] args) {
		ArbolBinario arbol = new ArbolBinario();
		int[] elementos = { 30, 20, 40, 10, 25, 35, 50 };
		for (int i : elementos) {
			arbol.insertar(i);
		}

		for (int i : elementos) {
			arbol.insertar(i);
		}
		System.out.println("---PREORDER---");
		arbol.preOrder(arbol.raiz);
		
		System.out.println("---postOrder---");
		arbol.postOrder(arbol.raiz);
		
		System.out.println("---inOrder---");
		arbol.inOrder(arbol.raiz);
	}
}