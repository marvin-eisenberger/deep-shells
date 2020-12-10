function preprocess_dataset(folder_data, file_extension)
    % preprocess_dataset - convert a folder of shapes to the .mat format required to train deep shells
    % folder_data: path to the raw shape data
    % file_extension: file extension of to the shape files [.off|.obj|.ply|.mat]

    file_arr = dir(folder_data + "/*" + file_extension);
    
    folder_processed = folder_data + "/processed/";
    
    if ~isfolder(folder_processed)
        mkdir(folder_processed);
    end
    
    for i = 1:length(file_arr)
        file_in = file_arr(i).folder + "/" + file_arr(i).name;
        
        [~, file_name, ~] = fileparts(file_in);
        file_out = folder_processed + file_name + ".mat";
        
        X = load_shape(file_in);
        
        save(file_out, "X");
    end
end