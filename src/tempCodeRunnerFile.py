# TODO: Remove dense code below vvv --- strictly for debugging purposes
    utils.close_plot()
    for i in range(3):  # Adjust to 3 examples
        # Get a batch of mixed and clean signals from the validation set
        batch = next(iter(val_loader))  # Assuming val_loader is an iterator
        mixed_data, clean_data, mixed_file, clean_file = batch  # Unpack the batch
        # Choose a random index to display
        random_idx = random.randint(0, mixed_data.shape[0] - 1)
        # Get the corresponding mixed and clean signals for this random index
        random_mixed_signal = mixed_data[random_idx:random_idx + 1]  # Shape: [1, 1, signal_length]
        random_clean_signal = clean_data[random_idx:random_idx + 1]  # Shape: [1, 1, signal_length]
        # Create a time array for plotting
        time = np.linspace(0, 1, int(random_mixed_signal.shape[-1]), endpoint=False)
        # Plot the random mixed signal
        name = f"Random Mixed Signal #{i + 1} from validation set"
        utils.plot_signal(time, random_mixed_signal.squeeze(), name)
        # Plot the corresponding clean (sine wave) signal
        name = f"Corresponding Clean Signal #{i + 1} (Sine Wave)"
        utils.plot_signal(time, random_clean_signal.squeeze(), name)
        # Plot the generator output
        generated_signal = generator(random_mixed_signal).detach().cpu().numpy().squeeze()
        name = f"Corresponding Denoised Signal #{i + 1}"
        utils.plot_signal(time, generated_signal, name)
        print(f"Displayed: mixed and clean signals #{i + 1}")
    # TODO: Remove above ^^^