class Producto:
    def __init__(self, nombre, precio, stock, marca, sede):
        self.nombre = nombre
        self.precio = precio
        self.stock = stock
        self.marca = marca
        self.sede = sede # Nueva propiedad sede 

    def restar_stock(self, cantidad):
        self.stock -= cantidad if cantidad <= self.stock else 0

#
producto = Producto("Smartphone", 800, 50, "Samsung", "Sede Central")
producto.restar_stock(5)
print(f"{producto.nombre} en {producto.sede} - Precio: {producto.precio}, Stock: {producto.stock}")
