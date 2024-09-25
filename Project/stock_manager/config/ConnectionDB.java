package config;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;

public class ConnectionDB {
	
	public Connection connect;
	public Statement state;
	public ResultSet resSet;
	public PreparedStatement prepState;

	public ConnectionDB() {

		// Buka database XAMPP
		try {
//			Class.forName("com.mysql.cj.jdbc.Driver");
			
												// buat nama database
			String url = "jdbc:mysql://localhost:3306/menu_database";
			String user = "root";
			String pass = "";
			
			connect = DriverManager.getConnection(url, user, pass);
			state = connect.createStatement();
					
		} catch (Exception e) {}
		
	}
	
	public ResultSet getAllData() {
		
		try {
//			resSet = state.executeQuery("SELECT * FROM menu_table");
			
															// buat nama tabel
			prepState = connect.prepareStatement("SELECT * FROM menu_table");
			resSet = prepState.executeQuery();
			
		} catch (SQLException e) {}
		
		return resSet;
		
	}
	
	public void insertData(String kode, String nama, double harga, int stok) {
		
		try {
			prepState = connect.prepareStatement("INSERT INTO menu_table (kode, nama, harga, stok) VALUES (?,?,?,?)");
			prepState.setString(1, kode);
			prepState.setString(2, nama);
			prepState.setDouble(3, harga);
			prepState.setInt(4, stok);
			
			prepState.execute();
			
		} catch (SQLException e) {}
		
	}
	
	public void updateData(String kode, double harga, int stok) {
		
		try {
			prepState = connect.prepareStatement("UPDATE menu_table SET harga = ?, stok = ? WHERE kode = ?");
			prepState.setDouble(1, harga);
			prepState.setInt(2, stok);
			prepState.setString(3, kode);
			
			prepState.execute();
			
		} catch (SQLException e) {}
		
	}
	
	public void deleteData(String kode) {
		
		try {
			prepState = connect.prepareStatement("DELETE FROM menu_table WHERE kode = ?");
			prepState.setString(1, kode);
			
			prepState.execute();
			
		} catch (SQLException e) {}
		
	}

}
