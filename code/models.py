import torch
import torch.nn as nn
import torch.nn.functional as F
import math

OUTPUT_RANGES = [[0.1, 1.5], [-1, 1], [1, 4], [1, 5], [0.1, 2.0]]

torch.manual_seed(1)

# Limit outputs as a set of sigmoid activation functions at the output layer.
def output_limits(x, use_cuda):
    # x shape: batch x output_indicies
    assert x.shape[1] == len(OUTPUT_RANGES), F"Input shape into the limit activation is {np.shape(x)}, \
                                                however the expected shape is ({np.shape(x)[0]}, {len(OUTPUT_RANGES)})"
    ranges = torch.FloatTensor(OUTPUT_RANGES)

    if use_cuda and torch.cuda.is_available():
        ranges = ranges.cuda()

    return torch.add(torch.mul(torch.sigmoid(x), (ranges[:, 1] - ranges[:, 0])), ranges[:, 0])

class GRUModel3(nn.Module): # Most original working GRU model
    def __init__(self, hidden_size=50, version=1.3, use_cuda=True):
        super(GRUModel3, self).__init__()
        self.name = "GRUModel"
        self.hidden_size = hidden_size
        self.version = version
        self.use_cuda = use_cuda
        self.value_norm = nn.Sequential(
            nn.BatchNorm1d(2, 0.0001, 0.99))
        self.interval_gru = nn.GRU(input_size=2, hidden_size=hidden_size, batch_first=True)
        self.detector_fc = nn.Linear(hidden_size*2 + 4, hidden_size)

        # Use CDAN-like method to concatenate unordered detector feature sets
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(p=0.1)
        self.overall_fc = nn.Linear(hidden_size, len(OUTPUT_RANGES))

        # Use GRU to concatenate unordered detector feature sets
        #self.detector_gru = nn.GRU(inpu第二季t_size=hidden_size, hidden_size=hidden_size*2, batch_first=True)
        #self.overall_fc = nn.Linear(hidden_size*2*2, len(OUTPUT_RANGES))

    def forward(self, values, locations):
        detector_count = values.shape[1]
        interval_count = values.shape[2]

        values = values.view(-1, interval_count, 2) # B*M (batch * detectors) x N (time intervals) x 2 (values)
        #values = values.permute(0, 2, 1)
        #values = self.value_norm(values)
        #values = values.permute(0, 2, 1)
        values, _ = self.interval_gru(values)
        values = torch.cat([torch.max(values, dim=1)[0], torch.mean(values, dim=1)], dim=1)  # max + mean

        locations = locations.view(-1, 4)

        per_detector = torch.cat((values, locations), 1)
        per_detector =self.detector_fc(per_detector)
        per_detector = per_detector.view(-1, detector_count, per_detector.shape[1])   # B (batch) x M (detectors) x hidden size
        per_detector = self.softmax(per_detector)

        all_detectors = torch.sum(per_detector, dim=1)

        #combined = combined.unsqueeze(0)
        #combined, _ = self.detector_gru(combined)
        #combined = torch.cat([torch.max(combined, dim=1)[0], torch.mean(combined, dim=1)], dim=1)  # max + mean

        output = self.overall_fc(all_detectors)

        return output_limits(output, use_cuda=self.use_cuda)

class GRUModel5(nn.Module): # Include lane number in training data
    def __init__(self, hidden_size=50, version=1.5, use_cuda=True, detector_is_in_order=True):
        super(GRUModel5, self).__init__()
        self.name = "double_GRU_model" if detector_is_in_order else "GRU_CDAN_model"
        self.hidden_size = hidden_size
        self.version = version
        self.use_cuda = use_cuda
        self.ordered_detectors = detector_is_in_order
        self.interval_gru = nn.GRU(input_size=2, hidden_size=hidden_size, batch_first=True)
        self.detector_fc = nn.Linear(hidden_size*2 + 5, hidden_size)

        if detector_is_in_order:
            # Use GRU to concatenate ordered detector feature sets
            self.detector_gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size*2, batch_first=True)
            self.overall_fc = nn.Linear(hidden_size*2*2, len(OUTPUT_RANGES))
        else:
            # Use CDAN-like method to concatenate unordered detector feature sets
            self.softmax = nn.Softmax(dim=2)
            self.overall_fc = nn.Linear(hidden_size, len(OUTPUT_RANGES))

    def forward(self, values, locations):
        detector_count = values.shape[1]
        interval_count = values.shape[2]

        values = values.view(-1, interval_count, 2) # B*M (batch * detectors) x N (time intervals) x 2 (values)
        values, _ = self.interval_gru(values)
        values = torch.cat([torch.max(values, dim=1)[0], torch.mean(values, dim=1)], dim=1)  # max + mean

        locations = locations.view(-1, 5)

        per_detector = torch.cat((values, locations), 1)
        per_detector =self.detector_fc(per_detector)
        per_detector = per_detector.view(-1, detector_count, per_detector.shape[1])   # B (batch) x M (detectors) x hidden size

        if self.ordered_detectors:
            combined, _ = self.detector_gru(per_detector)
            all_detectors = torch.cat([torch.max(combined, dim=1)[0], torch.mean(combined, dim=1)], dim=1)  # max + mean
        else:
            per_detector = self.softmax(per_detector)
            all_detectors = torch.sum(per_detector, dim=1)

        output = self.overall_fc(all_detectors)

        return output_limits(output, use_cuda=self.use_cuda)

class GRUModel6(nn.Module): # Combine location info with values as per-interval data before interval GRU
    def __init__(self, hidden_size=50, version=1.6, use_cuda=True, detector_is_in_order=True):
        super(GRUModel6, self).__init__()
        self.name = "double_GRU_model" if detector_is_in_order else "GRU_CDAN_model" 
        self.hidden_size = hidden_size
        self.version = version
        self.use_cuda = use_cuda
        self.ordered_detectors = detector_is_in_order
        self.interval_gru = nn.GRU(input_size=2+5, hidden_size=hidden_size, batch_first=True)
        self.detector_fc = nn.Linear(hidden_size*2, hidden_size)
        self.dropout = nn.Dropout(p=0.1)

        if detector_is_in_order:
            # Use GRU to concatenate ordered detector feature sets
            self.detector_gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size*2, batch_first=True)
            self.overall_fc = nn.Linear(hidden_size*2*2, len(OUTPUT_RANGES))
        else:
            # Use CDAN-like method to concatenate unordered detector feature sets
            self.softmax = nn.Softmax(dim=2)
            self.overall_fc = nn.Linear(hidden_size, len(OUTPUT_RANGES))

    def forward(self, values, locations):
        detector_count = values.shape[1]
        interval_count = values.shape[2]

        values = values.view(-1, interval_count, 2) # B*M (batch * detectors) x N (time intervals) x 2 (values)
        locations = locations.view(-1, 5).unsqueeze(dim=1).repeat(1, interval_count, 1)
        per_interval_loc_val = torch.cat((values, locations), dim=2)

        loc_val, _ = self.interval_gru(per_interval_loc_val)
        loc_val = torch.cat([torch.max(loc_val, dim=1)[0], torch.mean(loc_val, dim=1)], dim=1)  # max + mean

        per_detector = self.detector_fc(loc_val)
        per_detector = per_detector.view(-1, detector_count, per_detector.shape[1])   # B (batch) x M (detectors) x hidden size

        if self.ordered_detectors:
            combined, _ = self.detector_gru(per_detector)
            all_detectors = torch.cat([torch.max(combined, dim=1)[0], torch.mean(combined, dim=1)], dim=1)  # max + mean
        else:
            per_detector = self.softmax(per_detector)
            all_detectors = torch.sum(per_detector, dim=1)

        output = self.overall_fc(all_detectors)

        return output_limits(output, use_cuda=self.use_cuda)

class ScaledDotAttention(nn.Module):
    def __init__(self, hidden_size):
        super(ScaledDotAttention, self).__init__()

        self.hidden_size = hidden_size

        self.Q = nn.Linear(hidden_size, hidden_size*2)
        self.K = nn.Linear(hidden_size, hidden_size*2)
        self.V = nn.Linear(hidden_size, hidden_size*2)
        self.softmax = nn.Softmax(dim=1)
        self.scaling_factor = torch.rsqrt(torch.tensor(hidden_size*2, dtype= torch.float))

    def forward(self, queries, keys, values):
        """The forward pass of the scaled dot attention mechanism.

        Arguments:
            queries: The current decoder hidden state, 2D or 3D tensor. (batch_size x (k) x hidden_size)
            keys: The encoder hidden states for each step of the input sequence. (batch_size x seq_len x hidden_size)
            values: The encoder hidden states for each step of the input sequence. (batch_size x seq_len x hidden_size)

        Returns:
            context: weighted average of the values (batch_size x k x hidden_size)
            attention_weights: Normalized attention weights for each encoder hidden state. (batch_size x seq_len x 1)

            The output must be a softmax weighting over the seq_len annotations.
        """

        batch_size = queries.shape[0]
        expanded_queries = queries.view(batch_size, -1, self.hidden_size)
        q = self.Q(expanded_queries)
        k = self.K(keys)
        v = self.V(values)
        unnormalized_attention = torch.bmm(k, q.transpose(2,1)) * self.scaling_factor # batch_size x seq_len x (k)

        # for batch_i in range(batch_size):
        #     unnormalized_attention[batch_i, seq_lengths[batch_i]:, :] = -math.inf
        attention_weights = self.softmax(unnormalized_attention)   # batch_size x seq_len x (k)
        context = torch.bmm(attention_weights.transpose(2,1), v)
        return context

class GRUModel6_att(nn.Module): # Combine location info with values as per-interval data before interval GRU
    def __init__(self, hidden_size=50, version=1.61, use_cuda=True, detector_is_in_order=True):
        super(GRUModel6_att, self).__init__()
        self.name = "double_GRU_model" if detector_is_in_order else "GRU_CDAN_model" 
        self.hidden_size = hidden_size
        self.version = version
        self.use_cuda = use_cuda
        self.ordered_detectors = detector_is_in_order
        self.interval_gru = nn.GRU(input_size=2+5, hidden_size=hidden_size, batch_first=True)
        self.detector_fc = nn.Linear(hidden_size*2, hidden_size)
        self.dropout = nn.Dropout(p=0.1)

        if detector_is_in_order:
            # Use GRU to concatenate ordered detector feature sets
            self.detector_gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size*2, batch_first=True)
            self.overall_fc = nn.Linear(hidden_size*2*2, len(OUTPUT_RANGES))
        else:
            # Use CDAN-like method to concatenate unordered detector feature sets
            self.self_attentions = ScaledDotAttention(hidden_size=hidden_size)
            self.overall_fc = nn.Linear(hidden_size*2, len(OUTPUT_RANGES))

    def forward(self, values, locations):
        detector_count = values.shape[1]
        interval_count = values.shape[2]

        values = values.view(-1, interval_count, 2) # B*M (batch * detectors) x N (time intervals) x 2 (values)
        locations = locations.view(-1, 5).unsqueeze(dim=1).repeat(1, interval_count, 1)
        per_interval_loc_val = torch.cat((values, locations), dim=2)

        loc_val, _ = self.interval_gru(per_interval_loc_val)
        loc_val = torch.cat([torch.max(loc_val, dim=1)[0], torch.mean(loc_val, dim=1)], dim=1)  # max + mean

        per_detector = self.detector_fc(loc_val)
        per_detector = per_detector.view(-1, detector_count, per_detector.shape[1])   # B (batch) x M (detectors) x hidden size

        if self.ordered_detectors:
            combined, _ = self.detector_gru(per_detector)
            all_detectors = torch.cat([torch.max(combined, dim=1)[0], torch.mean(combined, dim=1)], dim=1)  # max + mean
        else:
            per_detector = self.self_attentions(per_detector, per_detector, per_detector)
            all_detectors = torch.sum(per_detector, dim=1)

        output = self.overall_fc(all_detectors)

        return output_limits(output, use_cuda=self.use_cuda)

class LSTMModel7(nn.Module): # added batch norm layers, used LSTM with more num_layers
    def __init__(self, hidden_size=50, version=1.7, use_cuda=True, detector_is_in_order=True):
        super(LSTMModel7, self).__init__()
        self.name = "double_LSTM_model" if detector_is_in_order else "LSTM_CDAN_model"
        self.hidden_size = hidden_size
        self.version = version
        self.use_cuda = use_cuda
        self.ordered_detectors = detector_is_in_order
        self.value_bn = nn.BatchNorm1d(2)
        self.interval_lstm = nn.LSTM(input_size=2, hidden_size=hidden_size, batch_first=True, num_layers=5)
        self.detector_fc = nn.Linear(hidden_size*2 + 5, hidden_size)
        self.dropout = nn.Dropout(p=0.1)
        if detector_is_in_order:
            # Use LSTM to concatenate ordered detector feature sets
            self.detector_lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size*2, batch_first=True, num_layers=5)
            self.overall_fc = nn.Linear(hidden_size*2*2, len(OUTPUT_RANGES))
        else:
            # Use CDAN-like method to concatenate unordered detector feature sets
            self.softmax = nn.Softmax(dim=2)
            self.overall_bn1 = nn.BatchNorm1d(hidden_size * 2)
            self.overall_fc1 = nn.Linear(hidden_size * 2, hidden_size)
            self.overall_bn2 = nn.BatchNorm1d(hidden_size)
            self.overall_fc2 = nn.Linear(hidden_size, hidden_size // 2)
            self.overall_bn3 = nn.BatchNorm1d(hidden_size // 2)
            self.overall_fc3 = nn.Linear(hidden_size // 2, len(OUTPUT_RANGES))

    def forward(self, values, locations):
        detector_count = values.shape[1]
        interval_count = values.shape[2]

        values = values.view(-1, 2)
        values = self.value_bn(values)
        values = values.view(-1, interval_count, 2) # B*M (batch * detectors) x N (time intervals) x 2 (values)
        values, _ = self.interval_lstm(values)
        values = torch.cat([torch.max(values, dim=1)[0], torch.mean(values, dim=1)], dim=1)  # max + mean

        locations = locations.view(-1, 5)

        per_detector = torch.cat((values, locations), 1)
        per_detector =self.detector_fc(per_detector)
        per_detector = per_detector.view(-1, detector_count, per_detector.shape[1])   # B (batch) x M (detectors) x hidden size

        if self.ordered_detectors:
            combined, _ = self.detector_lstm(per_detector)
            all_detectors = torch.cat([torch.max(combined, dim=1)[0], torch.mean(combined, dim=1)], dim=1)  # max + mean
            output = self.overall_fc(all_detectors)
        else:
            per_detector = self.softmax(per_detector)
            all_detectors = torch.cat([torch.max(per_detector, dim=1)[0], torch.mean(per_detector, dim=1)], dim=1)  # max + mean

            output = self.overall_fc1(self.overall_bn1(all_detectors))
            output = F.leaky_relu(output, negative_slope=0.2)
            output = self.dropout(output)
            output = self.overall_fc2(self.overall_bn2(output))
            output = F.leaky_relu(output, negative_slope=0.2)
            output = self.dropout(output)
            output = self.overall_fc3(self.overall_bn3(output))

        return output_limits(output, use_cuda=self.use_cuda)

class LSTMModel8(nn.Module): # added batch norm layers, used LSTM with more num_layers
    def __init__(self, hidden_size=50, version=1.8, use_cuda=True, detector_is_in_order=True):
        super(LSTMModel8, self).__init__()
        self.name = "double_LSTM_model" if detector_is_in_order else "LSTM_CDAN_model"
        self.hidden_size = hidden_size
        self.version = version
        self.use_cuda = use_cuda
        self.ordered_detectors = detector_is_in_order
        self.value_bn = nn.BatchNorm1d(2)
        self.interval_lstm = nn.LSTM(input_size=2, hidden_size=hidden_size, batch_first=True, num_layers=5)
        self.detector_fc = nn.Linear(hidden_size*2 + 5, hidden_size)
        self.dropout = nn.Dropout(p=0.1)
        if detector_is_in_order:
            # Use LSTM to concatenate ordered detector feature sets
            self.detector_lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size*2, batch_first=True, num_layers=5)
            self.overall_fc = nn.Linear(hidden_size*2*2, len(OUTPUT_RANGES))
        else:
            # Use CDAN-like method to concatenate unordered detector feature sets
            self.softmax = nn.Softmax(dim=1)
            self.overall_bn1 = nn.BatchNorm1d(hidden_size * 2)
            self.overall_fc1 = nn.Linear(hidden_size * 2, hidden_size)
            self.overall_bn2 = nn.BatchNorm1d(hidden_size)
            self.overall_fc2 = nn.Linear(hidden_size, hidden_size // 2)
            self.overall_bn3 = nn.BatchNorm1d(hidden_size // 2)
            self.overall_fc3 = nn.Linear(hidden_size // 2, len(OUTPUT_RANGES))

    def forward(self, values, locations, detector_counts):
        batch_count = values.shape[0]
        max_detector_count = values.shape[1]
        interval_count = values.shape[2]

        values = values.view(-1, values.shape[2], 2)
        preserved_indices = torch.hstack([torch.arange(detector_counts[i]) + max_detector_count * i for i in range(detector_counts.shape[0])]).to(values.device)
        values = values[preserved_indices, :]

        values = self.value_bn(values.view(-1, 2))
        values = values.view(-1, interval_count, 2) # B*M (batch * detectors) x N (time intervals) x 2 (values)
        values, _ = self.interval_lstm(values)
        values = torch.cat([torch.max(values, dim=1)[0], torch.mean(values, dim=1)], dim=1)  # max + mean

        locations = locations.view(-1, 5)
        locations = locations[preserved_indices, :]

        per_detector = torch.cat((values, locations), 1)
        per_detector =self.detector_fc(per_detector)

        full_per_detector = -float('Inf') * torch.ones([batch_count*max_detector_count, per_detector.shape[1]]).to(per_detector.device)
        full_per_detector[preserved_indices, :] = per_detector

        full_per_detector = full_per_detector.view(batch_count, max_detector_count, per_detector.shape[1])   # B (batch) x M (detectors) x hidden size

        if self.ordered_detectors:
            packed_detectors = torch.nn.utils.rnn.pack_padded_sequence(full_per_detector, detector_counts.cpu(), batch_first=True, enforce_sorted=False)
            packed_detectors, _ = self.detector_lstm(packed_detectors)
            padded_detectors, pack_sequence = torch.nn.utils.rnn.pad_packed_sequence(packed_detectors, batch_first=True, padding_value=0)

            all_detectors = torch.vstack([torch.cat([torch.max(padded_detectors[i, :pack_sequence[i]], dim=0)[0], \
                                                    torch.mean(padded_detectors[i, :pack_sequence[i]], dim=0)]) for i in range(batch_count)]) # max + mean

            output = self.overall_fc(all_detectors)
        else:
            all_detectors = torch.empty(0, full_per_detector.shape[2]*2).to(full_per_detector.device)
            for i in range(batch_count):
                batch_softmax = self.softmax(full_per_detector[i, :detector_counts[i], :])
                all_detectors = torch.vstack([all_detectors, torch.cat([torch.max(batch_softmax, dim=0)[0], torch.mean(batch_softmax, dim=0)])])

            output = self.overall_fc1(self.overall_bn1(all_detectors))
            output = F.leaky_relu(output, negative_slope=0.2)
            output = self.dropout(output)
            output = self.overall_fc2(self.overall_bn2(output))
            output = F.leaky_relu(output, negative_slope=0.2)
            output = self.dropout(output)
            output = self.overall_fc3(self.overall_bn3(output))

        return output_limits(output, use_cuda=self.use_cuda)