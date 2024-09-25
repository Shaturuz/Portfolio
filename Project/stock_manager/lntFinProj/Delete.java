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

public class Delete extends JFrame implements ActionListener{
	
	ConnectionDB db;
	
	JPanel topPanel, midPanel, botPanel,
	viewPanel,
	kodePanel1, kodePanel2,
	deleteBtnPanel, backBtnPanel;

	JLabel kodeLabel, viewLabel;
	JTextField kodeField;
	JButton deleteBtn, backBtn;

	Dimension dimensionSize = new Dimension(150, 25);
	
	void viewMenu()
	{
		setTitle("Menu Delete");
		setVisible(true);
		setSize(500, 400);
		setResizable(false);
		setLocationRelativeTo(null);
		setDefaultCloseOperation(EXIT_ON_CLOSE);
		
	}

	public Delete() {
		
		//Connection
		db = new ConnectionDB();

		// Top Panel
		topPanel = new JPanel(new GridLayout(2, 1));
		viewPanel = new JPanel();
		viewLabel = new JLabel("Delete");
		viewLabel.setFont(new Font("Arial", Font.BOLD, 30));
				
		viewPanel.add(viewLabel);
		topPanel.add(viewPanel);
		
		// Mid Panel;
		midPanel = new JPanel(new GridLayout(4, 1));
		kodePanel1 = new JPanel();
		kodePanel2 = new JPanel();
		
		kodeLabel = new JLabel("Kode");
		kodeField = new JTextField();
		kodeField.setPreferredSize(dimensionSize);
		
		kodePanel1.add(kodeLabel);
		kodePanel2.add(kodeField);
		
		midPanel.add(kodePanel1);
		midPanel.add(kodePanel2);
		
		//Bot Panel
		botPanel = new JPanel(new GridLayout(3, 2));
		deleteBtnPanel = new JPanel();
		backBtnPanel = new JPanel();
		
		deleteBtn = new JButton("Delete");
		deleteBtn.addActionListener(this);
		backBtn = new JButton("Back");
		backBtn.addActionListener(this);
		
		deleteBtnPanel.add(deleteBtn);
		backBtnPanel.add(backBtn);
		
		botPanel.add(deleteBtnPanel);
		botPanel.add(backBtnPanel);
		
		add(topPanel, BorderLayout.NORTH);
		add(midPanel, BorderLayout.CENTER);
		add(botPanel, BorderLayout.SOUTH);
		
		viewMenu();
	}

	@Override
	public void actionPerformed(ActionEvent e) {

		if(e.getSource() == deleteBtn) {
			
			String kode = kodeField.getText();
			
			db.deleteData(kode);
			
		}
		
		if(e.getSource() == backBtn) {
			
			this.dispose();
			new Main();
			
		}
		
	}


}
