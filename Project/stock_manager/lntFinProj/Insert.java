package lntFinProj;

import config.ConnectionDB;
import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.util.Random;

import javax.swing.ButtonGroup;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JPasswordField;
import javax.swing.JRadioButton;
import javax.swing.JTextField;

public class Insert extends JFrame implements ActionListener, KeyListener{
	
	ConnectionDB db;
	
	Random rand = new Random();
	
	JPanel topPanel, midPanel, botPanel,
			insPanel,
			kodePanel1, kodePanel2, namaPanel, hargaPanel, stokPanel, namaLPanel, hargaLPanel, stokLPanel,
			msgPanel, regisBtnPanel, backBtnPanel;

	JLabel insLabel, kodeLabel, kode, namaLabel, hargaLabel, stokLabel, msg;
	JTextField namaField, hargaField, stokField;
	JButton regisBtn, backBtn;
	
	Dimension dimensionSize = new Dimension(150, 25);
	
	void viewMenu()
	{
		setTitle("Menu Insert");
		setVisible(true);
		setSize(500, 400);
		setResizable(false);
		setLocationRelativeTo(null);
		setDefaultCloseOperation(EXIT_ON_CLOSE);
		
	}

	public Insert() {
		
		// Connection
		db = new ConnectionDB();
		
		// Top Panel
		topPanel = new JPanel(new GridLayout(2, 1));
		insPanel = new JPanel();
		insLabel = new JLabel("Insert");
		insLabel.setFont(new Font("Arial", Font.BOLD, 30));
		
		insPanel.add(insLabel);
		topPanel.add(insPanel);
		
		// Mid Panel
		midPanel = new JPanel(new GridLayout(5, 2));
		kodePanel1 = new JPanel();
		kodePanel2 = new JPanel();
		namaLPanel = new JPanel();
		hargaLPanel = new JPanel();
		stokLPanel = new JPanel();
		namaPanel = new JPanel();
		hargaPanel = new JPanel();
		stokPanel = new JPanel();
		
		kodeLabel = new JLabel("Kode");
		kode = new JLabel();
		namaLabel = new JLabel("Nama");
		hargaLabel = new JLabel("Harga");
		stokLabel = new JLabel("Stok");
		
		namaField = new JTextField();
		hargaField = new JTextField();
		stokField = new JTextField();
		namaField.addKeyListener(this);
		hargaField.addKeyListener(this);
		stokField.addKeyListener(this);
		
		kodePanel1.add(kodeLabel);
		kodePanel2.add(kode);
		namaLPanel.add(namaLabel);
		hargaLPanel.add(hargaLabel);
		stokLPanel.add(stokLabel);
		namaPanel.add(namaField);
		namaField.setPreferredSize(dimensionSize);
		hargaPanel.add(hargaField);
		hargaField.setPreferredSize(dimensionSize);
		stokPanel.add(stokField);
		stokField.setPreferredSize(dimensionSize);
		
		midPanel.add(namaLPanel);
		midPanel.add(namaPanel);
		midPanel.add(hargaLPanel);
		midPanel.add(hargaPanel);
		midPanel.add(stokLPanel);
		midPanel.add(stokPanel);
		midPanel.add(kodePanel1);
		midPanel.add(kodePanel2);
		
		// Bot Panel
		botPanel = new JPanel(new GridLayout(4, 1));
		regisBtnPanel = new JPanel();
		backBtnPanel = new JPanel();
		msgPanel = new JPanel();
		
		msg = new JLabel();
		regisBtn = new JButton("Register");
		regisBtn.addActionListener(this);
		backBtn = new JButton("Back");
		backBtn.addActionListener(this);
		
		msgPanel.add(msg);
		regisBtnPanel.add(regisBtn);
		backBtnPanel.add(backBtn);
		
		botPanel.add(msgPanel);
		botPanel.add(regisBtnPanel);
		botPanel.add(backBtnPanel);
		
		add(topPanel, BorderLayout.NORTH);
		add(midPanel, BorderLayout.CENTER);
		add(botPanel, BorderLayout.SOUTH);
		viewMenu();
	}
	
	String generateKode() {
		
		String base = "PD-";
		
//		int r = rand.nextInt(9);
		
		char a = (char) ('0' + rand.nextInt(9));
		char b = (char) ('0' + rand.nextInt(9));
		char c = (char) ('0' + rand.nextInt(9));
		
		String kode = base + a + b + c;
		
//		System.out.println(kode);
		return kode;
	}

	int cek = 0;
	
	@Override
	public void actionPerformed(ActionEvent e) {
		
		if(e.getSource() == regisBtn) {
			
			if(!namaField.getText().equals("") && !hargaField.getText().equals("") && 
					!stokField.getText().equals("")) {
				
				String nama = namaField.getText();
				
				Double harga = (double) 0;
				Integer stok = 0;
				try {
					harga = Double.valueOf(hargaField.getText());
					stok = Integer.valueOf(stokField.getText());
				} catch (NumberFormatException e1) {}
				
				
				String kodeValue;
				
				if (cek == 0) {
					
					kodeValue = generateKode();
					kode.setText(kodeValue);
					cek = 1;
				}
				
				kodeValue = kode.getText();
				
				namaField.setEditable(false);
				hargaField.setEditable(false);
				stokField.setEditable(false);
				
//				System.out.println(kodeValue);
//				System.out.println(nama);
//				System.out.println(harga);
//				System.out.println(stok);
				
				db.insertData(kodeValue, nama, harga, stok);
				
				msg.setText("Data berhasil dimasukkan");
				
//				try {
//					regisBtn.addActionListener(null);
//					Thread.sleep(1500);
//				} catch (InterruptedException e1) {}
				
//				this.dispose();
//				new Main();
				
			}
			else {
				msg.setText("Data masih kosong");
			}
			
			
		}
		
		if(e.getSource() == backBtn) {
			
			this.dispose();
			new Main();
			
		}
		
	}

	@Override
	public void keyTyped(KeyEvent e) {
		
		
	}

	@Override
	public void keyPressed(KeyEvent e) {
		
		
	}

	@Override
	public void keyReleased(KeyEvent e) {
		
		
	}
	
}
