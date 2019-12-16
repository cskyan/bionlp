/*************************************************************************
    > Copyright (C) 2013-2019 by Caspar. All rights reserved.
    > File Name: a.java
    > Author: Shankai Yan
    > E-mail: shankai.yan@nih.gov
    > Created Time: 2019-12-16 11:43:16
************************************************************************/

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.io.FileWriter;
import java.io.BufferedWriter;
import java.io.IOException;
import java.util.Arrays;

import py4j.GatewayServer;
import gov.nih.nlm.nls.skr.GenericObject;


public class MTIApp {

    private String username;
    private String password;
    private String email;

    public MTIApp(String user, String passwd, String e_mail) {
        username = user;
        password = passwd;
        email = e_mail;
    }

    public String get_result(String fpath) {
        GenericObject genericObj = new GenericObject(username, password);
        genericObj.setField("Email_Address", email);
        genericObj.setFileField("UpLoad_File", fpath);
        genericObj.setField("Batch_Command", "MTI -opt1L_DCMS -E");
        genericObj.setField("BatchNotes", "MTI batch submission");
        genericObj.setField("SilentEmail", true);
        try {
            String result = genericObj.handleSubmission();
            return result;
        } catch (RuntimeException ex) {
            System.err.println(String.format("Encounter errors when processing file %s", fpath));
            ex.printStackTrace();
        }
        return "";
    }

    public String[] get_fnames(String input_path) {
        File in_dir = new File(input_path);
        if (! in_dir.exists()) throw new NullPointerException(String.format("The input path %s does not exists!", input_path));
        File file = new File(input_path);
        String[] fnames = file.list();
        return fnames;
    }

    public boolean write_file(String str, String fpath) {
    		try {
    			File file = new File(fpath);
    			if (!file.exists()) {
    				file.createNewFile();
    			}
    			FileWriter fileWriter = new FileWriter(file.getAbsolutePath(), true);
    			BufferedWriter bw = new BufferedWriter(fileWriter);
    			bw.write(str);
    			bw.close();
    			return true;
    		} catch (IOException e) {
    			e.printStackTrace();
    		}
    		return false;
    }

    public void get_results(String input_path, String output_path, String[] fnames, int start, int end) {
        File in_dir = new File(input_path);
        if (! in_dir.exists()) throw new NullPointerException(String.format("The input path %s does not exists!", input_path));
        File out_dir = new File(output_path);
        if (! out_dir.exists()) out_dir.mkdir();
        for (int i = start; i < Math.min(end, fnames.length); i++) {
            String fpath = fnames[i];
            Path in_path = Paths.get(input_path, fpath);
            Path out_path = Paths.get(output_path, fpath);
            File out_file = new File(out_path.toString());
            if (out_file.exists()) {
                System.out.println(out_path.toString() + " exists!");
                continue;
            }
            String result = get_result(in_path.toString());
            write_file(result, out_path.toString());
            System.out.println(fpath + " processed.");
        }
    }

    public void batch(String input_path, String output_path, int batch_size) {
        String[] fnames = get_fnames(input_path);
        int batch_num = fnames.length / batch_size;
        int remainder = fnames.length % batch_size;
        int[] batch_sizes;
        if (remainder > 0) {
            batch_sizes = new int[batch_num+1];
            Arrays.fill(batch_sizes, 0, batch_num, batch_size);
            batch_sizes[batch_num] = remainder;
        } else {
            batch_sizes = new int[batch_num];
            Arrays.fill(batch_sizes, 0, batch_num, batch_size);
        }

        for (int i = 0, start = 0; i < batch_sizes.length; i++) {
            int end = start + batch_sizes[i];
            get_results(input_path, output_path, fnames, start, end);
            start = end;
        }
    }

    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("Please input your username, password and email address separated by white space!");
            System.exit(1);
        }
        GatewayServer gatewayServer = new GatewayServer(new MTIApp(args[0], args[1], args.length>2 ? args[2] : "abc@example.com"));
        gatewayServer.start();
        System.out.println("MTI Gateway Server Started");
    }
}
