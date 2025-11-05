
public class ArbolBinario {
	Nodo raiz;
	public ArbolBinario() {
		raiz = null;
	}
	
	public Nodo insertarRecursivo(Nodo raiz, int valor) {
		if (raiz == null) {
			raiz = new Nodo(valor);
		}
		if (valor < raiz.valor) {
			raiz.izquierdo=insertarRecursivo(raiz.izquierdo,valor);
		}
		else if(valor > raiz.valor) {
			raiz.derecho= insertarRecursivo(raiz.derecho, valor);
		}
		return raiz;
	}
	public void insertar (int valor) {
		raiz= insertarRecursivo(raiz, valor);
	}
	
	public void preOrder(Nodo nodo) {
		if (nodo != null) {
			System.out.println(nodo.valor+" ");
			preOrder(nodo.izquierdo);
			preOrder(nodo.derecho);
		}
	}
	
	public void inOrder(Nodo nodo) {
		if (nodo != null) {
			inOrder(nodo.izquierdo);
			System.out.println(nodo.valor+" ");
			inOrder(nodo.derecho);
		}
	}
	public void postOrder(Nodo nodo) {
		if (nodo != null) {
			postOrder(nodo.izquierdo);
			postOrder(nodo.derecho);
			System.out.println(nodo.valor+" ");
		}
	}
}
