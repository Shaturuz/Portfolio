package lntFinProj;

import config.ConnectionDB;
import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;

public class Update extends JFrame implements ActionListener{
	
	ConnectionDB db;
	
	JPanel topPanel, midPanel, botPanel,
	viewPanel,
	kodePanel1, kodePanel2, hargaPanel, stokPanel, hargaLPanel, stokLPanel,
	updateBtnPanel, backBtnPanel;

	JLabel kodeLabel, viewLabel, hargaLabel, stokLabel;
	JTextField kodeField, hargaField, stokField;
	JButton updateBtn, backBtn;

	Dimension dimensionSize = new Dimension(150, 25);
	
	void viewMenu()
	{
		setTitle("Menu Update");
		setVisible(true);
		setSize(500, 400);
		setResizable(false);
		setLocationRelativeTo(null);
		setDefaultCloseOperation(EXIT_ON_CLOSE);
		
	}

	public Update() {

		// Connection
		db = new ConnectionDB();
		
		// Top Panel
		topPanel = new JPanel(new GridLayout(2, 1));
		viewPanel = new JPanel();
		viewLabel = new JLabel("Update");
		viewLabel.setFont(new Font("Arial", Font.BOLD, 30));
				
		viewPanel.add(viewLabel);
		topPanel.add(viewPanel);
		
		// Mid Panel;
		midPanel = new JPanel(new GridLayout(3, 2));
		kodePanel1 = new JPanel();
		kodePanel2 = new JPanel();
		hargaPanel = new JPanel();
		stokPanel = new JPanel();
		hargaLPanel = new JPanel();
		stokLPanel = new JPanel();
		
		kodeLabel = new JLabel("Kode");
		kodeField = new JTextField();
		kodeField.setPreferredSize(dimensionSize);
		hargaLabel = new JLabel("Harga");
		stokLabel = new JLabel("Stok");
		hargaField = new JTextField();
		hargaField.setPreferredSize(dimensionSize);
		stokField = new JTextField();
		stokField.setPreferredSize(dimensionSize);
		
		kodePanel1.add(kodeLabel);
		kodePanel2.add(kodeField);
		hargaLPanel.add(hargaLabel);
		hargaPanel.add(hargaField);
		stokLPanel.add(stokLabel);
		stokPanel.add(stokField);
		
		midPanel.add(kodePanel1);
		midPanel.add(kodePanel2);
		midPanel.add(hargaLPanel);
		midPanel.add(hargaPanel);
		midPanel.add(stokLPanel);
		midPanel.add(stokPanel);
		
		//Bot Panel
		botPanel = new JPanel(new GridLayout(3, 2));
		updateBtnPanel = new JPanel();
		backBtnPanel = new JPanel();
		
		updateBtn = new JButton("Update");
		updateBtn.addActionListener(this);
		backBtn = new JButton("Back");
		backBtn.addActionListener(this);
		
		updateBtnPanel.add(updateBtn);
		backBtnPanel.add(backBtn);
		
		botPanel.add(updateBtnPanel);
		botPanel.add(backBtnPanel);
		
		add(topPanel, BorderLayout.NORTH);
		add(midPanel, BorderLayout.CENTER);
		add(botPanel, BorderLayout.SOUTH);
		
		viewMenu();
	}

	@Override
	public void actionPerformed(ActionEvent e) {

		if(e.getSource() == updateBtn) {
			
			if(!kodeField.getText().equals("") && !hargaField.getText().equals("") && !stokField.getText().equals("")) {
				
				String kode = kodeField.getText();
				double harga = Double.valueOf(hargaField.getText());
				int stok = Integer.valueOf(stokField.getText());
				
				db.updateData(kode, harga, stok);
				
			}
			
		}
		
		if(e.getSource() == backBtn) {
			
			this.dispose();
			new Main();
			
		}
		
	}

}
