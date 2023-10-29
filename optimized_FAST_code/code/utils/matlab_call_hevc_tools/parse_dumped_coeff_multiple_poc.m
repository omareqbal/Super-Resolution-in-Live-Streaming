function [PU_all, res_luma_all] = parse_dumped_coeff_multiple_poc(file_path, num_frames)

% Parse Multiple Frames. Parse Intra and MV.
% tic
fid = fopen(file_path, 'r');
lines = textscan(fid,'%s','delimiter','\n');
lines = lines{1};
fclose(fid);

PU_all = cell(1,num_frames);
res_luma_all = cell(1,num_frames);

% tid = -1;

res_frame_all = cell(1, 3);
res_frame_all{1} = struct('x', {[]}, 'y', {[]}, 'w', {[]}, 'residual', {[]});
res_frame_all{2} = struct('x', {[]}, 'y', {[]}, 'w', {[]}, 'residual', {[]});
res_frame_all{3} = struct('x', {[]}, 'y', {[]}, 'w', {[]}, 'residual', {[]});
PU_frame_all = struct('intra', {[]}, 'x', {[]}, 'y', {[]}, ...
    'w', {[]}, 'h', {[]}, ...
    'mv_x', {[]}, 'mv_y', {[]}, 't', {[]}, 't_r', {[]}, ...
    'luma_mode', {[]});

PU_all{1} = PU_frame_all;
res_luma_all{1} = res_frame_all{1};

num_lines = size(lines, 1);


poc_pos = zeros(1, num_frames);
frames_found = 0;

line_segments_all = cell(1, num_lines);
for i = 1:num_lines
    line_segments = strsplit(lines{i}, {':', ',', ' '});
    line_segments_all{i} = line_segments;
    if strcmp(line_segments{1},'POC')
        frames_found = frames_found+1;
        poc_pos(frames_found) = i;
    end
end

line_segments_frames = cell(1, num_frames-1);

for f_idx = 1:num_frames-1
    frame_st = poc_pos(f_idx);
    frame_en = poc_pos(f_idx+1);
    line_segments_frames{f_idx} = line_segments_all(frame_st+1:frame_en-1);
end

parfor f_idx = 1:num_frames-1
    line_segments_frame = line_segments_frames{f_idx};

    tid = -1;
    res_frame_all = cell(1, 3);
    res_frame_all{1} = struct('x', {[]}, 'y', {[]}, 'w', {[]}, 'residual', {[]});
    res_frame_all{2} = struct('x', {[]}, 'y', {[]}, 'w', {[]}, 'residual', {[]});
    res_frame_all{3} = struct('x', {[]}, 'y', {[]}, 'w', {[]}, 'residual', {[]});
    PU_frame_all = struct('intra', {[]}, 'x', {[]}, 'y', {[]}, ...
        'w', {[]}, 'h', {[]}, ...
        'mv_x', {[]}, 'mv_y', {[]}, 't', {[]}, 't_r', {[]}, ...
        'luma_mode', {[]});

    num_lines = size(line_segments_frame,2);
    for i = 1:num_lines
        % Judge whether the line starts with one of the keywords.
        % line_segments = strsplit(line, {':', ',', ' '});
        line_segments = line_segments_frame{i};
        switch lower(line_segments{1})
            case 'coeff'
                nums = cellstr2num(line_segments(2:end));
                % qper = nums(1);
                % qrem = nums(2);
                w = nums(3);
                % list = nums(4);
                n = w * w;
                if length(nums) ~= 4 + 4 * n
                    fprintf('Last line reached!');
                end
                % scaling = nums((4 + 1):(4 + n));
                % levels = nums((4 + n + 1):(4 + 2 * n));
                try
                    % coeffs = nums((4 + 2 * n + 1):(4 + 2 * n + n));
                    residuals = nums((4 + 2 * n + n + 1):...
                    (4 + 2 * n + n + n));
                catch
                    db_var = 1;
                end
                
                
                res_frame_all{tid}(end).w = w;
                res_frame_all{tid}(end).residual = residuals;
                % if all(levels == 0)
                %     %                 fprintf('Encountering an all-zero TU!\n');
                %     assert(all(coeffs == 0) && all(residuals == 0), ...
                %         'A TU with all zero scalings has non zero coeffs & residuals');
                % end
                tid = -1;
                %             fprintf('Parse a TU coeff of size %dx%d!\n', w, w);
            case 'tu'
                nums = cellstr2num(line_segments(2:end));
                x = nums(1);
                y = nums(2);
                text = nums(3);
                % intra = nums(4);
                
                %             fprintf('Locate a TU at (%d, %d, %d, %d)!\n', ...
                %                 x, y, text, intra);
                
                % Initialize a TU
                res_struct = struct('x', {x}, 'y', {y}, 'w', {[]}, ...
                    'residual', {[]});
                tid = max(text, 1);
                res_frame_all{tid}(length(res_frame_all{tid}) + 1) = res_struct;
            case 'intra'
                prop_map = parse_prop(line_segments(2:end));
                PU_frame = struct('intra', {1}, 'x', {prop_map('x')}, ...
                    'y', {prop_map('y')}, ...
                    'w', {prop_map('w')}, 'h', {prop_map('h')}, ...
                    'mv_x', {[]}, 'mv_y', {[]}, 't', {prop_map('t')}, ...
                    't_r', {[]}, 'luma_mode', {prop_map('luma_mode')});
                PU_frame_all(length(PU_frame_all) + 1) = PU_frame;
            case 'mv'
                prop_map = parse_prop(line_segments(2:end));
                PU_frame = struct('intra', {0}, 'x', {prop_map('x')}, ...
                    'y', {prop_map('y')}, ...
                    'w', {prop_map('w')}, 'h', {prop_map('h')}, ...
                    'mv_x', {prop_map('mv_x')}, 'mv_y', {prop_map('mv_y')},...
                    't', {prop_map('t')}, 't_r', {prop_map('t_r')}, ...
                    'luma_mode', {[]});
                PU_frame_all(length(PU_frame_all) + 1) = PU_frame;
            case 'poc'
                fprintf('POC found! Start a new frame!\n');
                
            otherwise
                fprintf('Unparsed statement: %s\n', line_segments{1});
        end
    end
    res_luma_all{f_idx+1} = res_frame_all{1};
    PU_all{f_idx+1} = PU_frame_all;

end
% tmp = toc;
% fprintf('Parse dump.txt - %f\n', tmp);
end

function prop_map = parse_prop(line)
prop_map = containers.Map();
for i = 1:length(line)
    vars = strsplit(line{i}, '=');
    for t = 1:(length(vars) - 1)
        prop_map(vars{t}) = str2num(vars{end});
    end 
end

end