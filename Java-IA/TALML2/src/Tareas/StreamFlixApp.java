package Tareas;

import java.awt.Color;
import java.awt.EventQueue;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.util.ArrayList;

import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.JTextField;
import javax.swing.border.EmptyBorder;
import javax.swing.table.DefaultTableModel;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class StreamFlixApp extends JFrame {

    private static final long serialVersionUID = 1L;
    private JPanel contentPane;
    private JTable grilla;
    
    // Campos para inputs
    private JTextField txt_edadUsuario;
    private JTextField txt_duracionPeli;
    private JComboBox<String> cbx_genero;
    private JComboBox<String> cbx_premios;
    private JLabel JL_result;

    // Variables Weka
    private Instances data;
    private ArrayList<Attribute> atributos;
    private Classifier modelJ;

    public static void main(String[] args) {
        EventQueue.invokeLater(new Runnable() {
            public void run() {
                try {
                    StreamFlixApp frame = new StreamFlixApp();
                    frame.setVisible(true);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });
    }

    public StreamFlixApp() {
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setBounds(100, 100, 918, 400);
        contentPane = new JPanel();
        contentPane.setBorder(new EmptyBorder(5, 5, 5, 5));
        setContentPane(contentPane);
        contentPane.setLayout(null);

        JPanel panel = new JPanel();
        panel.setBounds(0, 0, 902, 360);
        contentPane.add(panel);
        panel.setLayout(null);

        // Tabla de datos históricos
        JScrollPane scrollPane = new JScrollPane();
        scrollPane.setBounds(10, 11, 689, 250);
        panel.add(scrollPane);
        grilla = new JTable();
        scrollPane.setViewportView(grilla);

        // --- Controles de UI ---
        JLabel lblEdad = new JLabel("Edad Usuario:");
        lblEdad.setBounds(712, 21, 100, 14);
        panel.add(lblEdad);

        txt_edadUsuario = new JTextField();
        txt_edadUsuario.setBounds(712, 37, 150, 20);
        panel.add(txt_edadUsuario);

        JLabel lblGenero = new JLabel("Género Película:");
        lblGenero.setBounds(712, 68, 100, 14);
        panel.add(lblGenero);

        cbx_genero = new JComboBox<>();
        cbx_genero.setBounds(712, 84, 150, 22);
        // Coinciden con la DB
        cbx_genero.addItem("Sci-Fi");
        cbx_genero.addItem("Action");
        cbx_genero.addItem("Drama");
        cbx_genero.addItem("Comedy");
        cbx_genero.addItem("Romance");
        panel.add(cbx_genero);

        JLabel lblDuracion = new JLabel("Duración (min):");
        lblDuracion.setBounds(712, 116, 100, 14);
        panel.add(lblDuracion);

        txt_duracionPeli = new JTextField();
        txt_duracionPeli.setBounds(712, 132, 150, 20);
        panel.add(txt_duracionPeli);

        JLabel lblPremios = new JLabel("¿Tiene Premios?:");
        lblPremios.setBounds(712, 165, 100, 14);
        panel.add(lblPremios);

        cbx_premios = new JComboBox<>();
        cbx_premios.setBounds(712, 181, 150, 22);
        cbx_premios.addItem("TRUE"); // Mapeado a 1 en DB o boolean
        cbx_premios.addItem("FALSE");
        panel.add(cbx_premios);

        JButton btn_prediction = new JButton("Analizar y Recomendar");
        btn_prediction.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                predecirRecomendacion();
            }
        });
        btn_prediction.setBounds(712, 230, 180, 30);
        panel.add(btn_prediction);

        JL_result = new JLabel("Esperando datos...");
        JL_result.setFont(new java.awt.Font("Tahoma", java.awt.Font.BOLD, 12));
        JL_result.setBounds(10, 280, 500, 20);
        panel.add(JL_result);

        // Cargar datos y entrenar modelo al iniciar
        cargarEntrenamiento();
    }

    // Bloque 1: Carga de datos, configuración de Weka y Entrenamiento
    public void cargarEntrenamiento() {
        String url = "jdbc:mysql://localhost/StreamFlixDB";
        String user = "root";
        String password = ""; 

        try {
            Connection cn = DriverManager.getConnection(url, user, password);
            
            // Query que une Usuarios, Peliculas e Interacciones
            // Usamos el rating para determinar la CLASE (Si gustó o no)
            String sql = "SELECT u.age, m.genre, m.duration_minutes, m.is_award_winner, i.rating " +
                         "FROM UserInteractions i " +
                         "JOIN Users u ON i.user_id = u.user_id " +
                         "JOIN Movies m ON i.movie_id = m.movie_id";
            
            PreparedStatement ps = cn.prepareStatement(sql);
            ResultSet rs = ps.executeQuery();

            DefaultTableModel model = new DefaultTableModel(null,
                    new String[] { "Edad", "Género", "Duración", "Premios", "Clase(Recomendar)" });

            // Definición de Atributos Weka
            atributos = new ArrayList<Attribute>();
            
            // 1. Edad (Numérico)
            atributos.add(new Attribute("age"));
            
            // 2. Género (Nominal)
            ArrayList<String> valGenero = new ArrayList<>();
            valGenero.add("Sci-Fi"); valGenero.add("Action"); valGenero.add("Drama");
            valGenero.add("Comedy"); valGenero.add("Romance");
            atributos.add(new Attribute("genre", valGenero));

            // 3. Duración (Numérico)
            atributos.add(new Attribute("duration"));

            // 4. Premios (Nominal)
            ArrayList<String> valPremios = new ArrayList<>();
            valPremios.add("TRUE");
            valPremios.add("FALSE");
            atributos.add(new Attribute("awards", valPremios));

            // 5. CLASE: Recomendar (Nominal - Objetivo del Árbol)
            ArrayList<String> valClase = new ArrayList<>();
            valClase.add("SI"); // Rating >= 4.0
            valClase.add("NO"); // Rating < 4.0
            atributos.add(new Attribute("recomendar", valClase));

            // Crear set de instancias
            data = new Instances("StreamFlixData", atributos, 0);
            data.setClassIndex(data.numAttributes() - 1); // La última columna es la clase

            while (rs.next()) {
                double rating = rs.getDouble("rating");
                // Lógica: Si rating >= 4.0, la clase es SI, sino NO
                String claseRecomendacion = (rating >= 4.0) ? "SI" : "NO";
                String tienePremios = rs.getBoolean("is_award_winner") ? "TRUE" : "FALSE";

                // Llenar Weka Instance
                Instance instance = new DenseInstance(data.numAttributes());
                instance.setValue(atributos.get(0), rs.getInt("age"));
                instance.setValue(atributos.get(1), rs.getString("genre"));
                instance.setValue(atributos.get(2), rs.getInt("duration_minutes"));
                instance.setValue(atributos.get(3), tienePremios);
                instance.setValue(atributos.get(4), claseRecomendacion); 
                
                instance.setDataset(data);
                data.add(instance);

                // Llenar JTable visual
                model.addRow(new Object[] { 
                    rs.getInt("age"), rs.getString("genre"), 
                    rs.getInt("duration_minutes"), tienePremios, claseRecomendacion 
                });
            }
            grilla.setModel(model);

            // Construir el Árbol J48
            modelJ = new J48();
            modelJ.buildClassifier(data);
            JL_result.setText("Modelo entrenado con éxito. Listo para predecir.");

        } catch (Exception e) {
            e.printStackTrace();
            JL_result.setText("Error de conexión o entrenamiento: " + e.getMessage());
        }
    }

    // Bloque 2: Predicción basada en inputs del usuario
    public void predecirRecomendacion() {
        try {
            if (modelJ == null) return;

            Instance muestra = new DenseInstance(data.numAttributes());
            muestra.setDataset(data); // Importante vincular al header

            // Obtener valores de la UI
            muestra.setValue(atributos.get(0), Integer.parseInt(txt_edadUsuario.getText()));
            muestra.setValue(atributos.get(1), cbx_genero.getSelectedItem().toString());
            muestra.setValue(atributos.get(2), Integer.parseInt(txt_duracionPeli.getText()));
            muestra.setValue(atributos.get(3), cbx_premios.getSelectedItem().toString());
            // La clase (índice 4) se deja vacía ("missing") para predecir

            // Clasificar
            double indice = modelJ.classifyInstance(muestra);
            String prediccion = data.classAttribute().value((int) indice);

            if (prediccion.equals("SI")) {
                JL_result.setForeground(new Color(0, 150, 0)); // Verde
                JL_result.setText("RESULTADO: ¡Película RECOMENDADA para este usuario!");
            } else {
                JL_result.setForeground(Color.RED);
                JL_result.setText("RESULTADO: No recomendar (Probablemente no le guste).");
            }

        } catch (Exception e) {
            JL_result.setText("Error en predicción: Verifique los datos numéricos.");
        }
    }
}