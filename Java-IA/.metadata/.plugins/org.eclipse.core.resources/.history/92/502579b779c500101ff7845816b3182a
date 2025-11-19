package Ejercicios;

import java.io.File;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

public class DSUtil {

    public void crearDataset() throws Exception {
        // Obtener datos
        ResultSet rs = new DAO().obtenerDatos();

        // Definir atributos
        ArrayList<Attribute> atributos = new ArrayList<>();
        atributos.add(new Attribute("edad"));
        atributos.add(new Attribute("asistencia"));
        atributos.add(new Attribute("promedio"));

        ArrayList<String> valoresConducta = new ArrayList<>();
        valoresConducta.add("buena");
        valoresConducta.add("regular");
        valoresConducta.add("mala");
        atributos.add(new Attribute("conducta", valoresConducta));

        ArrayList<String> valoresApoyo = new ArrayList<>();
        valoresApoyo.add("si");
        valoresApoyo.add("no");
        atributos.add(new Attribute("apoyo_familiar", valoresApoyo));

        ArrayList<String> valoresAbandono = new ArrayList<>();
        valoresAbandono.add("si");
        valoresAbandono.add("no");
        atributos.add(new Attribute("abandono", valoresAbandono));

        // Crear dataset
        Instances data = new Instances("dataset", atributos, 0);
        data.setClassIndex(data.numAttributes() - 1);

        // Llenar dataset con los datos del ResultSet
        while (rs.next()) {
            Instance instance = new DenseInstance(data.numAttributes());
            instance.setValue(atributos.get(0), rs.getInt("edad"));
            instance.setValue(atributos.get(1), rs.getInt("asistencia"));
            instance.setValue(atributos.get(2), rs.getDouble("promedio"));
            instance.setValue(atributos.get(3), rs.getString("conducta"));
            instance.setValue(atributos.get(4), rs.getString("apoyo_familiar"));
            instance.setValue(atributos.get(5), rs.getString("abandono"));
            instance.setDataset(data);
            data.add(instance);
        }

        // Guardar en ARFF
        ArffSaver arff = new ArffSaver();
        arff.setInstances(data);
        arff.setFile(new File("data.arff"));
        arff.writeBatch();

        rs.close();
    }

    public static void main(String[] args) {
        try {
            new DSUtil().crearDataset();
            System.out.println("ARFF generado correctamente.");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
