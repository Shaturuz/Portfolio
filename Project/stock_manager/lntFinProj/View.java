package lntFinProj;

import config.ConnectionDB;
import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.sql.SQLException;
import java.util.Iterator;
import java.util.Vector;

import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.JTextField;

public class View extends JFrame implements ActionListener{
	
	ConnectionDB db;
	
	JPanel topPanel, labelPanel, midPanel, valuePanel, botPanel,
	viewPanel,
	kodeLPanel, kodePanel, namaLPanel, namaPanel, hargaLPanel, hargaPanel, stokLPanel, stokPanel,
	panel1, panel2, panel3, panel4,
	backBtnPanel;

	JLabel viewLabel, kodeLabel, namaLabel, hargaLabel, stokLabel;
	JTextArea kodeValue, namaValue, hargaValue, stokValue;
	JButton backBtn;
	JScrollPane scroll1;

	Dimension dimensionSize = new Dimension(150, 25);
	
	void viewMenu()
	{
		setTitle("Menu View");
		setVisible(true);
		setSize(500, 400);
		setResizable(false);
		setLocationRelativeTo(null);
		setDefaultCloseOperation(EXIT_ON_CLOSE);
		
	}

	public View() {
		
		// Connection
		db = new ConnectionDB();
		db.getAllData();
		
//		Vector<String> data = new Vector<>();
		Vector<String> kode = new Vector<>();
		Vector<String> nama = new Vector<>();
		Vector<Double> harga = new Vector<>();
		Vector<Integer> stok = new Vector<>();
		String kodeV = "";
		String namaV = "";
		double hargaV = 0;
		int stokV = 0;
		
		try {
			while(db.resSet.next()) {
				
				kodeV = db.resSet.getString(1);
				namaV = db.resSet.getString(2);
				hargaV = db.resSet.getDouble(3);
				stokV = db.resSet.getInt(4);
				
				kode.add(kodeV);
				nama.add(namaV);
				harga.add(hargaV);
				stok.add(stokV);
				
			}
		} catch (SQLException e) {}
		
		// Top Panel
		topPanel = new JPanel(new GridLayout(2, 4));
		viewPanel = new JPanel();
		viewLabel = new JLabel("View");
		viewLabel.setFont(new Font("Arial", Font.BOLD, 30));
		labelPanel = new JPanel(new GridLayout(1, 1));
		kodeLPanel = new JPanel();
		namaLPanel = new JPanel();
		hargaLPanel = new JPanel();
		stokLPanel = new JPanel();
		
		kodeLabel = new JLabel("Kode");
		namaLabel = new JLabel("Nama");
		hargaLabel = new JLabel("Harga");
		stokLabel = new JLabel("Stok");
		
		kodeLPanel.add(kodeLabel);
		namaLPanel.add(namaLabel);
		hargaLPanel.add(hargaLabel);
		stokLPanel.add(stokLabel);
		
		viewPanel.add(viewLabel);
		labelPanel.add(kodeLPanel);
		labelPanel.add(namaLPanel);
		labelPanel.add(hargaLPanel);
		labelPanel.add(stokLPanel);
		
		topPanel.add(viewPanel);
		topPanel.add(labelPanel);
		
		// Mid Panel
		midPanel = new JPanel(new GridLayout(2, 4));
		valuePanel = new JPanel();
		kodePanel = new JPanel();
//		namaLPanel = new JPanel();
		namaPanel = new JPanel();
		hargaPanel = new JPanel();
//		stokLPanel = new JPanel();
		stokPanel = new JPanel();
		panel1 = new JPanel();
		panel2 = new JPanel();
		panel3 = new JPanel();
		panel4 = new JPanel();
		
		
		
		kodeValue = new JTextArea(kode.size(), 7);
		kodeValue.setEditable(false);
		
		namaValue = new JTextArea(nama.size(), 10);
		namaValue.setEditable(false);
		
		hargaValue = new JTextArea(harga.size(), 10);
		hargaValue.setEditable(false);
		
		stokValue = new JTextArea(stok.size(), 10);
		stokValue.setEditable(false);
		scroll1 = new JScrollPane(valuePanel);
		
		String data1 = "";
		String data2 = "";
		String data3 = "";
		String data4 = "";
		
		for (int i = 0; i < kode.size(); i++) {
			
			data1 = data1 + kode.get(i) + "\n";
			data2 = data2 + nama.get(i) + "\n";
			data3 = data3 + harga.get(i).toString() + "\n";
			data4 = data4 + stok.get(i) + "\n";
			
		}
		
//		System.out.println(data1);
//		System.out.println(data2);
//		System.out.println(data3);
//		System.out.println(data4);
		
		kodeValue.setText(data1);
		namaValue.setText(data2);
		hargaValue.setText(data3);
		stokValue.setText(data4);
		
		kodePanel.add(kodeValue);
		namaPanel.add(namaValue);
		hargaPanel.add(hargaValue);
		stokPanel.add(stokValue);
		
		valuePanel.add(kodePanel);
		valuePanel.add(namaPanel);
		valuePanel.add(hargaPanel);
		valuePanel.add(stokPanel);
//		midPanel.add(kodeLPanel);
//		midPanel.add(namaLPanel);
//		midPanel.add(hargaLPanel);
//		midPanel.add(stokLPanel);
		midPanel.add(scroll1);
		midPanel.add(panel1);
		midPanel.add(panel2);
		midPanel.add(panel3);
		midPanel.add(panel4);
		
		
		
		// Bot Panel
		botPanel = new JPanel(new GridLayout(2, 1));
		backBtnPanel = new JPanel();
		
		backBtn = new JButton("Back");
		backBtn.addActionListener(this);
		
		backBtnPanel.add(backBtn);
	
		botPanel.add(backBtnPanel);
		
		getContentPane().add(scroll1);
		add(topPanel, BorderLayout.NORTH);
//		add(midPanel, BorderLayout.CENTER);
		add(scroll1, BorderLayout.CENTER);
		add(botPanel, BorderLayout.SOUTH);
		viewMenu();
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		
		if(e.getSource() == backBtn) {
			
			this.dispose();
			new Main();
			
		}
		
	}

}
