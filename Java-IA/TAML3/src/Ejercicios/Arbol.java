package Ejercicios;

import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Arbol {
    public static void main(String[] args) {
        try {
            new DSUtil().crearDataset();
            Instances data = DataSource.read("data.arff");

            // Establecer la variable objetivo
            data.setClassIndex(data.numAttributes() - 1);

            //Crear y entrenar el modelo J48
            J48 model = new J48();
            model.buildClassifier(data);

            System.out.println(model);

            //Realizar una prediccion
            Instance nuevo = data.firstInstance();
            double prediccion = model.classifyInstance(nuevo);
            String clase = data.classAttribute().value((int) prediccion);

            System.out.println("Predicción para el primer registro: " + clase);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
