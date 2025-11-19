package Ejercicios;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;

public class DAO {
	public ResultSet obtenerDatos() throws SQLException {
		String url="jdbc:mysql://localhost/colegio";
		String user="root";
		String password="";
		Connection cn= DriverManager.getConnection(url, user, password);
		String consulta="select edad, asistencia, promedio, "
		+ "conducta, apoyo_familiar, abandono "
		+ "from estudiantes_riesgo";
		PreparedStatement ps=cn.prepareStatement(consulta);
		ResultSet rs=ps.executeQuery();
		return rs;
		}
}
